"""
foam_packaging_optimizer.py
============================
Drop-test impact physics model for protective packaging design.

Engineering model
------------------
1. FREE-FALL IMPACT VELOCITY
       v = sqrt(2 * g * h)

2. KINETIC ENERGY AT IMPACT
       KE = 1/2 * m * v^2

3. CUSHION "ENERGY METHOD" (the standard first-pass hand-calc used in
   packaging engineering before validating against a manufacturer's
   ASTM D1596 static cushion curve):

   The foam is idealized as absorbing the impact energy over a stopping
   stroke  d = t * eps_max , where t is foam thickness and eps_max is the
   maximum usable strain before the foam "bottoms out" (goes rigid).

       F_avg = KE / d                       (average retarding force)
       G_avg = F_avg / (m * g)              (average deceleration, in g's)
       G_peak = G_avg * k_shape             (peak deceleration)

   k_shape ("peak-to-average factor") captures how peaky vs. flat a given
   foam's real force-deflection curve is -- linear-elastic foams (PE) are
   flatter (k close to 1.3-1.5); more brittle/crushable foams (EPS,
   molded pulp) spike harder near bottom-out (k close to 1.6-2.0).
   These are representative textbook values, NOT a substitute for a real
   cushion curve from the foam supplier.

4. MINIMUM THICKNESS FOR A GIVEN FRAGILITY RATING
   Every product has a fragility rating G_f (the "G&F" rating, i.e. the
   max deceleration it can survive without damage -- typically supplied
   by testing or by class: 40G very fragile electronics, 60G fragile,
   85-100G semi-rugged, >100G rugged). Solving the energy method for
   thickness:

       G_allow_avg = G_f / k_shape
       F_allow      = G_allow_avg * g * m
       d_min        = KE / F_allow
       t_min        = d_min / eps_max

   A safety factor (default 1.25x) is then applied to cover real-world
   variability (foam aging, temperature, off-corner/off-edge drops,
   compression set over shipping duration).

Limitations (stated explicitly, not hidden):
  - This is a lumped-mass, constant-average-force approximation. It is the
    standard "quick-sizing" method taught in packaging engineering courses
    and is adequate for prototyping / comparative design.
  - Final designs should be verified against the foam manufacturer's
    published dynamic cushion curves (G vs. static stress, per ASTM D1596)
    for the exact material, thickness, and drop height, and confirmed with
    physical drop testing (ASTM D4169 / ISTA procedures).
"""

from dataclasses import dataclass
import csv
import math

G_EARTH = 9.81  # m/s^2


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass
class Product:
    name: str
    mass_kg: float
    fragility_g: float        # max survivable deceleration, in g's
    footprint_area_m2: float  # cushion contact (bearing) area


@dataclass
class FoamMaterial:
    name: str
    max_strain: float          # eps_max: usable strain before bottom-out
    peak_to_avg_factor: float  # k_shape
    typical_static_stress_kpa: tuple  # (low, high) recommended design range
    notes: str = ""


MATERIAL_LIBRARY = {
    "EPS (expanded polystyrene)": FoamMaterial(
        name="EPS (expanded polystyrene)",
        max_strain=0.65,
        peak_to_avg_factor=1.8,
        typical_static_stress_kpa=(3.5, 14.0),
        notes="Rigid, cost-effective, brittle at low temp, single-impact biased.",
    ),
    "PE foam (polyethylene, e.g. Ethafoam)": FoamMaterial(
        name="PE foam (polyethylene, e.g. Ethafoam)",
        max_strain=0.70,
        peak_to_avg_factor=1.4,
        typical_static_stress_kpa=(3.0, 20.0),
        notes="Resilient, good multi-impact recovery, moderate cost.",
    ),
    "PU foam (polyurethane, flexible)": FoamMaterial(
        name="PU foam (polyurethane, flexible)",
        max_strain=0.60,
        peak_to_avg_factor=2.0,
        typical_static_stress_kpa=(1.0, 8.0),
        notes="Very soft/low stress, good for light products, compresses easily.",
    ),
    "Molded pulp / recycled fiber": FoamMaterial(
        name="Molded pulp / recycled fiber",
        max_strain=0.45,
        peak_to_avg_factor=2.1,
        typical_static_stress_kpa=(5.0, 25.0),
        notes="Sustainable, stiffer, less strain range, poor multi-impact recovery.",
    ),
}


# ---------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------

def impact_velocity(drop_height_m: float, g: float = G_EARTH) -> float:
    """Free-fall impact velocity, v = sqrt(2 g h)."""
    return math.sqrt(2 * g * drop_height_m)


def kinetic_energy(mass_kg: float, velocity_m_s: float) -> float:
    """Translational kinetic energy at impact."""
    return 0.5 * mass_kg * velocity_m_s ** 2


def peak_g_for_thickness(product: Product, foam: FoamMaterial,
                          drop_height_m: float, thickness_m: float) -> float:
    """Predicted peak deceleration (in g's) for a given foam thickness."""
    v = impact_velocity(drop_height_m)
    ke = kinetic_energy(product.mass_kg, v)
    stroke_m = thickness_m * foam.max_strain
    if stroke_m <= 0:
        return float("inf")
    f_avg = ke / stroke_m
    g_avg = f_avg / (product.mass_kg * G_EARTH)
    return g_avg * foam.peak_to_avg_factor


def required_thickness(product: Product, foam: FoamMaterial,
                        drop_height_m: float,
                        safety_factor: float = 1.25) -> dict:
    """Minimum foam thickness (per cushion pad, i.e. thickness on ONE side
    of the impact axis) to keep peak deceleration at/under the product's
    fragility rating, plus supporting intermediate results."""
    v = impact_velocity(drop_height_m)
    ke = kinetic_energy(product.mass_kg, v)

    g_allow_avg = product.fragility_g / foam.peak_to_avg_factor
    f_allow = g_allow_avg * G_EARTH * product.mass_kg
    stroke_min_m = ke / f_allow
    t_min_m = stroke_min_m / foam.max_strain
    t_min_safe_m = t_min_m * safety_factor

    static_stress_kpa = (f_allow / product.footprint_area_m2) / 1000.0

    return {
        "impact_velocity_m_s": v,
        "kinetic_energy_J": ke,
        "min_thickness_mm": t_min_m * 1000,
        "recommended_thickness_mm": t_min_safe_m * 1000,
        "design_static_stress_kPa": static_stress_kpa,
        "stress_in_material_range": (
            foam.typical_static_stress_kpa[0]
            <= static_stress_kpa
            <= foam.typical_static_stress_kpa[1]
        ),
    }


# ---------------------------------------------------------------------
# Reporting / sweep utilities
# ---------------------------------------------------------------------

def compare_materials(product: Product, drop_height_m: float,
                       safety_factor: float = 1.25) -> list:
    """Run required_thickness() for every material in the library."""
    rows = []
    for mat in MATERIAL_LIBRARY.values():
        res = required_thickness(product, mat, drop_height_m, safety_factor)
        rows.append({"material": mat.name, **res})
    rows.sort(key=lambda r: r["recommended_thickness_mm"])
    return rows


def g_vs_thickness_curve(product: Product, foam: FoamMaterial,
                          drop_height_m: float,
                          t_min_mm: float = 3, t_max_mm: float = 80,
                          steps: int = 60) -> list:
    """Sweep thickness and return [(thickness_mm, peak_G), ...] for plotting."""
    pts = []
    for i in range(steps):
        t_mm = t_min_mm + (t_max_mm - t_min_mm) * i / (steps - 1)
        g_peak = peak_g_for_thickness(product, foam, drop_height_m, t_mm / 1000.0)
        pts.append((t_mm, g_peak))
    return pts


def print_report(product: Product, drop_height_m: float,
                  safety_factor: float = 1.25) -> list:
    v = impact_velocity(drop_height_m)
    ke = kinetic_energy(product.mass_kg, v)

    print("=" * 72)
    print(f"DROP-TEST IMPACT ANALYSIS -- {product.name}")
    print("=" * 72)
    print(f"Mass                 : {product.mass_kg:.3f} kg")
    print(f"Fragility rating     : {product.fragility_g:.0f} G")
    print(f"Cushion contact area : {product.footprint_area_m2 * 1e4:.1f} cm^2")
    print(f"Drop height          : {drop_height_m:.2f} m")
    print("-" * 72)
    print(f"Impact velocity       v = sqrt(2*g*h)  = {v:.3f} m/s")
    print(f"Kinetic energy at impact  KE = 1/2 m v^2 = {ke:.3f} J")
    print("-" * 72)
    print(f"{'Material':38s}{'t_min (mm)':>12s}{'t_recommended (mm)':>22s}")
    rows = compare_materials(product, drop_height_m, safety_factor)
    for r in rows:
        flag = "" if r["stress_in_material_range"] else "  (check stress range)"
        print(f"{r['material']:38s}{r['min_thickness_mm']:12.1f}"
              f"{r['recommended_thickness_mm']:22.1f}{flag}")
    print("=" * 72)
    print(f"Best (thinnest viable) option: {rows[0]['material']}")
    print(f"  -> Recommended thickness: {rows[0]['recommended_thickness_mm']:.1f} mm "
          f"per side (includes {safety_factor}x safety factor)")
    print(f"  -> Design static stress on foam: "
          f"{rows[0]['design_static_stress_kPa']:.2f} kPa")
    print("=" * 72)
    return rows


def export_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_cushion_curves(product: Product, drop_height_m: float,
                         out_path: str = "cushion_curves.png") -> None:
    """Optional chart: peak G vs. thickness for every material, with the
    product's fragility limit overlaid. Requires matplotlib; skipped
    gracefully if it isn't installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping chart (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed"]

    for (name, foam), color in zip(MATERIAL_LIBRARY.items(), colors):
        curve = g_vs_thickness_curve(product, foam, drop_height_m,
                                      t_min_mm=3, t_max_mm=90, steps=80)
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        ax.plot(xs, ys, label=foam.name, color=color, linewidth=2)

        res = required_thickness(product, foam, drop_height_m)
        ax.scatter([res["recommended_thickness_mm"]], [product.fragility_g],
                   color=color, zorder=5, s=40, edgecolor="white")

    ax.axhline(product.fragility_g, color="crimson", linestyle="--",
               linewidth=1.5, label=f"Fragility limit ({product.fragility_g} G)")
    ax.set_xlabel("Foam thickness per side (mm)")
    ax.set_ylabel("Predicted peak deceleration (G)")
    ax.set_title(f"Cushion Curves — {product.mass_kg} kg product, "
                 f"{drop_height_m} m drop\n"
                 f"(dots = recommended thickness incl. safety factor)")
    ax.set_ylim(0, min(400, max(product.fragility_g * 4, 200)))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    demo_product = Product(
        name="Consumer electronics unit (e.g. small appliance / speaker)",
        mass_kg=1.8,
        fragility_g=60,          # typical "fragile" G&F rating
        footprint_area_m2=0.020, # ~ 4 cushion pads, 5cm x 10cm each combined
    )
    demo_drop_height_m = 0.9  # 3 ft -- standard ISTA parcel drop-test height

    result_rows = print_report(demo_product, demo_drop_height_m, safety_factor=1.25)
    export_csv(result_rows, "material_comparison.csv")
    plot_cushion_curves(demo_product, demo_drop_height_m, "cushion_curves.png")