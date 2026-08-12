import numpy as np
import streamlit as st
import sympy as sp
from scipy.integrate import quad
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import plotly.graph_objects as go


# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(
    page_title="Disk Method Explorer",
    page_icon="🟦",
    layout="wide",
)

# Keep the app deliberately HTML/CSS-free for maximum Streamlit Cloud compatibility.


# ============================================================
# Parsing helpers
# ============================================================
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
SAFE_FUNCS = {
    "pi": sp.pi,
    "e": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
}


def safe_parse_x(expr_str: str):
    x = sp.Symbol("x", real=True)
    local_dict = {"x": x, **SAFE_FUNCS}
    expr = parse_expr(
        expr_str,
        local_dict=local_dict,
        transformations=TRANSFORMS,
        evaluate=True,
    )
    return x, sp.simplify(expr)


def as_array(values, template):
    if np.isscalar(values):
        return np.full_like(template, float(values), dtype=float)
    return np.asarray(values, dtype=float)


# ============================================================
# Geometry helpers
# ============================================================
def cylinder_mesh_x(x0, x1, radius, axis_y=0.0, n_theta=72):
    """Closed cylinder for one disk, axis parallel to x."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    y_ring = axis_y + radius * np.cos(theta)
    z_ring = radius * np.sin(theta)

    x = np.concatenate([
        np.full(n_theta, x0),
        np.full(n_theta, x1),
        [x0, x1],
    ])
    y = np.concatenate([y_ring, y_ring, [axis_y, axis_y]])
    z = np.concatenate([z_ring, z_ring, [0.0, 0.0]])

    front_center = 2 * n_theta
    back_center = 2 * n_theta + 1

    I, J, K = [], [], []

    for k in range(n_theta):
        kn = (k + 1) % n_theta

        # side wall
        I.extend([k, k])
        J.extend([kn, n_theta + kn])
        K.extend([n_theta + kn, n_theta + k])

        # front cap
        I.append(front_center)
        J.append(kn)
        K.append(k)

        # back cap
        I.append(back_center)
        J.append(n_theta + k)
        K.append(n_theta + kn)

    return x, y, z, I, J, K


def exact_surface_mesh(xs, radii, axis_y=0.0, n_theta=96):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X = np.tile(xs, (theta.size, 1))
    TH = np.tile(theta.reshape(-1, 1), (1, xs.size))
    RR = np.tile(radii, (theta.size, 1))
    Y = axis_y + RR * np.cos(TH)
    Z = RR * np.sin(TH)
    return X, Y, Z


def rim_polyline(x_positions, radii, axis_y=0.0, n_theta=80):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X, Y, Z = [], [], []
    for xpos, radius in zip(x_positions, radii):
        X.extend(np.full(theta.size, xpos).tolist())
        Y.extend((axis_y + radius * np.cos(theta)).tolist())
        Z.extend((radius * np.sin(theta)).tolist())
        X.append(None)
        Y.append(None)
        Z.append(None)
    return X, Y, Z


# ============================================================
# Color palettes
# ============================================================
def palette_pack(name: str):
    palettes = {
        "Ocean blue": {
            "disk_colors": ["#2563EB", "#3B82F6", "#60A5FA"],
            "rim": "rgba(20, 64, 145, 0.82)",
            "curve": "#111827",
            "region": "rgba(59, 130, 246, 0.22)",
            "sample": "#16A34A",
            "partition": "rgba(37, 99, 235, 0.38)",
            "axis2d": "#EF4444",
            "axis3d": "#475569",
            "smooth": "#93C5FD",
        },
        "Emerald teal": {
            "disk_colors": ["#0F766E", "#14B8A6", "#5EEAD4"],
            "rim": "rgba(15, 93, 86, 0.82)",
            "curve": "#111827",
            "region": "rgba(20, 184, 166, 0.22)",
            "sample": "#F59E0B",
            "partition": "rgba(13, 148, 136, 0.38)",
            "axis2d": "#DC2626",
            "axis3d": "#475569",
            "smooth": "#99F6E4",
        },
        "Sunset coral": {
            "disk_colors": ["#EA580C", "#F97316", "#FB923C"],
            "rim": "rgba(154, 52, 18, 0.82)",
            "curve": "#111827",
            "region": "rgba(249, 115, 22, 0.22)",
            "sample": "#0EA5E9",
            "partition": "rgba(234, 88, 12, 0.38)",
            "axis2d": "#BE123C",
            "axis3d": "#475569",
            "smooth": "#FDBA74",
        },
        "Violet": {
            "disk_colors": ["#6D28D9", "#8B5CF6", "#A78BFA"],
            "rim": "rgba(76, 29, 149, 0.82)",
            "curve": "#111827",
            "region": "rgba(139, 92, 246, 0.22)",
            "sample": "#22C55E",
            "partition": "rgba(124, 58, 237, 0.38)",
            "axis2d": "#E11D48",
            "axis3d": "#475569",
            "smooth": "#C4B5FD",
        },
        "Teaching rainbow": {
            "disk_colors": [
                "#2563EB", "#06B6D4", "#14B8A6", "#22C55E",
                "#EAB308", "#F97316", "#EF4444", "#8B5CF6",
            ],
            "rim": "rgba(51, 65, 85, 0.72)",
            "curve": "#111827",
            "region": "rgba(99, 102, 241, 0.18)",
            "sample": "#16A34A",
            "partition": "rgba(99, 102, 241, 0.32)",
            "axis2d": "#DC2626",
            "axis3d": "#475569",
            "smooth": "#CBD5E1",
        },
    }
    base = palettes[name]
    return {
        **base,
        "scene_bg": "#F8FAFC",
        "paper_bg": "#FFFFFF",
        "font": "#1E293B",
        "grid": "#DCE3EC",
        "minor_grid": "#EEF2F7",
    }


# ============================================================
# Sidebar controls — simplified for students
# ============================================================
with st.sidebar:
    st.header("Disk Method Controls")
    st.caption("Build a solid by revolving the region between y = f(x) and y = c around the horizontal line y = c.")

    f_str = st.text_input("Curve  y = f(x)", value="sqrt(x)")

    st.subheader("Interval")
    c1, c2 = st.columns(2)
    with c1:
        a = st.number_input("a", value=0.0, format="%.4f")
    with c2:
        b = st.number_input("b", value=4.0, format="%.4f")

    axis_c = st.number_input("Axis of rotation  y = c", value=0.0, format="%.4f")

    st.divider()
    st.subheader("Approximation")
    n_disks = st.slider("Number of disks, n", min_value=2, max_value=60, value=8, step=1)
    sample_rule = st.selectbox(
        "Radius sampled at",
        ["Midpoint", "Left endpoint", "Right endpoint"],
        index=0,
    )

    with st.expander("Visual options", expanded=False):
        palette_name = st.selectbox(
            "Color palette",
            ["Ocean blue", "Emerald teal", "Sunset coral", "Violet", "Teaching rainbow"],
            index=0,
        )
        visual_gap_pct = st.slider(
            "Visual separation between disks",
            min_value=0,
            max_value=24,
            value=12,
            step=1,
            help="Display-only spacing. The volume calculation still uses the full mathematical thickness Δx.",
        )
        show_rims = st.toggle("Emphasize disk edges", value=True)
        show_true_surface = st.toggle("Overlay true smooth solid", value=False)
        show_partitions = st.toggle("Show partition lines in 2D", value=True)
        n_theta = st.slider("Disk roundness", 40, 128, 80, 8)

colors = palette_pack(palette_name)


# ============================================================
# Validate and evaluate function
# ============================================================
if np.isclose(a, b):
    st.error("Choose an interval with a ≠ b.")
    st.stop()

if b < a:
    a, b = b, a
    st.info("I swapped the endpoints so that a < b.")

try:
    x, f_expr = safe_parse_x(f_str)
    f_num = sp.lambdify(x, f_expr, modules=["numpy"])
except Exception as exc:
    st.error(f"I could not parse f(x). Details: {exc}")
    st.stop()

xs = np.linspace(a, b, 800)
try:
    f_vals = as_array(f_num(xs), xs)
except Exception as exc:
    st.error(f"f(x) could not be evaluated on [{a}, {b}]. Details: {exc}")
    st.stop()

if not np.all(np.isfinite(f_vals)):
    st.error("f(x) is not finite everywhere on the chosen interval. Adjust f(x), a, or b.")
    st.stop()

true_radii = np.abs(f_vals - axis_c)


# ============================================================
# Disk Riemann sum
# ============================================================
edges = np.linspace(a, b, n_disks + 1)
dx = (b - a) / n_disks
lefts = edges[:-1]
rights = edges[1:]
centers = (lefts + rights) / 2

if sample_rule == "Left endpoint":
    x_star = lefts
elif sample_rule == "Right endpoint":
    x_star = rights
else:
    x_star = centers

try:
    f_star = as_array(f_num(x_star), x_star)
except Exception as exc:
    st.error(f"The sample points could not be evaluated. Details: {exc}")
    st.stop()

if not np.all(np.isfinite(f_star)):
    st.error("At least one disk sample point gives a non-finite value.")
    st.stop()

radii = np.abs(f_star - axis_c)
disk_volumes = np.pi * radii**2 * dx
approx_volume = float(np.sum(disk_volumes))


def integrand(t):
    y_val = float(np.asarray(f_num(t)))
    return np.pi * (y_val - axis_c) ** 2


try:
    exact_volume_num, quad_err = quad(integrand, a, b, limit=300)
except Exception as exc:
    st.error(f"The volume integral failed numerically. Details: {exc}")
    st.stop()

try:
    sym_volume = sp.integrate(sp.pi * (f_expr - axis_c) ** 2, (x, a, b))
    if sym_volume.has(sp.Integral):
        sym_volume = None
    else:
        sym_volume = sp.simplify(sym_volume)
except Exception:
    sym_volume = None

abs_error = abs(approx_volume - exact_volume_num)
rel_error = abs_error / abs(exact_volume_num) if not np.isclose(exact_volume_num, 0.0) else np.nan


# ============================================================
# Header
# ============================================================
st.title("Disk Method Explorer")
st.caption("2D region  →  disk approximation  →  3D solid of revolution")
st.markdown(
    "Change the function, interval, axis of rotation, or number of disks. "
    "Then drag the 3D model to inspect the solid from any angle."
)


# ============================================================
# STEP 1 — GeoGebra-style 2D panel
# ============================================================
st.divider()
st.header("1. Explore the 2D region")
st.caption("The shaded region is the part that will rotate around the red axis.")

fig2d = go.Figure()

# Region fill
fig2d.add_trace(
    go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([f_vals, np.full_like(xs, axis_c)[::-1]]),
        fill="toself",
        mode="lines",
        line=dict(width=0),
        fillcolor=colors["region"],
        name="Region",
        hoverinfo="skip",
    )
)

# Function curve
fig2d.add_trace(
    go.Scatter(
        x=xs,
        y=f_vals,
        mode="lines",
        line=dict(width=3.5, color=colors["curve"]),
        name="y = f(x)",
        hovertemplate="x = %{x:.4g}<br>y = %{y:.4g}<extra></extra>",
    )
)

# Axis of rotation
fig2d.add_trace(
    go.Scatter(
        x=[a, b],
        y=[axis_c, axis_c],
        mode="lines",
        line=dict(width=3, color=colors["axis2d"]),
        name="axis y = c",
        hoverinfo="skip",
    )
)

# Sample points
fig2d.add_trace(
    go.Scatter(
        x=x_star,
        y=f_star,
        mode="markers",
        marker=dict(
            size=9,
            color=colors["sample"],
            line=dict(width=1.2, color="#FFFFFF"),
        ),
        name="radius samples",
        customdata=np.column_stack([np.arange(1, n_disks + 1), radii]),
        hovertemplate=(
            "Disk %{customdata[0]:.0f}<br>"
            "x* = %{x:.5g}<br>"
            "radius = %{customdata[1]:.5g}<extra></extra>"
        ),
    )
)

if show_partitions:
    px, py = [], []
    for edge in edges:
        try:
            y_edge = float(np.asarray(f_num(edge)))
        except Exception:
            continue
        px.extend([edge, edge, None])
        py.extend([axis_c, y_edge, None])
    fig2d.add_trace(
        go.Scatter(
            x=px,
            y=py,
            mode="lines",
            line=dict(width=1.2, color=colors["partition"]),
            name="partitions",
            hoverinfo="skip",
        )
    )

# GeoGebra-inspired coordinate plane
all_y = np.concatenate([f_vals, np.array([axis_c])])
y_min = float(np.min(all_y))
y_max = float(np.max(all_y))
y_pad = max(0.35, 0.12 * max(1e-9, y_max - y_min))
x_pad = max(0.35, 0.06 * abs(b - a))

axis_common = dict(
    showgrid=True,
    gridcolor=colors["grid"],
    gridwidth=1,
    zeroline=True,
    zerolinecolor="#64748B",
    zerolinewidth=2,
    showline=True,
    linecolor="#CBD5E1",
    linewidth=1,
    ticks="outside",
    tickcolor="#94A3B8",
    tickfont=dict(size=12, color="#475569"),
    title_font=dict(size=14, color="#334155"),
    fixedrange=False,
)

fig2d.update_layout(
    height=610,
    margin=dict(l=35, r=25, t=50, b=35),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    hovermode="closest",
    dragmode="pan",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(
        **axis_common,
        title="x",
        range=[a - x_pad, b + x_pad],
        minor=dict(showgrid=True, gridcolor=colors["minor_grid"], gridwidth=1),
    ),
    yaxis=dict(
        **axis_common,
        title="y",
        range=[y_min - y_pad, y_max + y_pad],
        scaleanchor="x",
        scaleratio=1,
        minor=dict(showgrid=True, gridcolor=colors["minor_grid"], gridwidth=1),
    ),
    uirevision="2d-region-view",
)

st.plotly_chart(
    fig2d,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    },
)
st.caption("2D panel: drag to pan · scroll to zoom · double-click to reset the view.")


# ============================================================
# STEP 2 — Math, with built-in Streamlit metrics
# ============================================================
st.divider()
st.header("2. Connect the picture to the disk formula")
st.caption("Each disk has thickness Δx and radius |f(x*) − c|.")

m1, m2, m3, m4 = st.columns(4, gap="large")
with m1:
    st.caption("Δx")
    st.markdown(f"### {dx:.5g}")
with m2:
    st.caption("Disk approximation")
    st.markdown(f"### {approx_volume:.7g}")
with m3:
    st.caption("Actual volume")
    st.markdown(f"### {exact_volume_num:.7g}")
with m4:
    st.caption("Relative error")
    error_text = "—" if np.isnan(rel_error) else f"{100 * rel_error:.4g}%"
    st.markdown(f"### {error_text}")

formula_col, exact_col = st.columns(2, gap="large")
with formula_col:
    st.markdown("**Disk-method Riemann sum**")
    st.latex(r"\Delta x=\frac{b-a}{n}")
    st.latex(r"V_n=\sum_{i=1}^{n}\pi\,[f(x_i^*)-c]^2\,\Delta x")
    st.caption(f"Sampling rule: {sample_rule}. Here n = {n_disks} and Δx = {dx:.6g}.")

with exact_col:
    st.markdown("**Exact volume**")
    if sym_volume is not None:
        st.latex(r"V=\pi\int_a^b [f(x)-c]^2\,dx=" + sp.latex(sym_volume))
    else:
        st.latex(r"V=\pi\int_a^b [f(x)-c]^2\,dx")
        st.caption(f"Numerical integration error estimate: ±{quad_err:.2g}")


# ============================================================
# STEP 3 — Large 3D teaching panel
# ============================================================
st.divider()
st.header(f"3. Explore the 3D approximation with {n_disks} disks")
st.caption("Drag the solid to rotate it. The small gaps make the individual disks easier to see; they do not change the mathematics.")

fig3d = go.Figure()

if show_true_surface:
    Xg, Yg, Zg = exact_surface_mesh(
        xs,
        true_radii,
        axis_y=axis_c,
        n_theta=max(96, n_theta),
    )
    fig3d.add_trace(
        go.Surface(
            x=Xg,
            y=Yg,
            z=Zg,
            surfacecolor=np.zeros_like(Xg),
            colorscale=[[0, colors["smooth"]], [1, colors["smooth"]]],
            showscale=False,
            opacity=0.16,
            hoverinfo="skip",
            name="True solid",
        )
    )

visual_gap = (visual_gap_pct / 100.0) * dx
disk_centers_vis = []

# Softer, more natural lighting than the original high-specular red material.
max_radius = max(float(np.max(radii)), 1e-6)
scene_span = max(abs(b - a), 2 * max_radius, 1.0)
light_pos = dict(
    x=float(a - 1.2 * scene_span),
    y=float(axis_c - 1.6 * scene_span),
    z=float(2.3 * scene_span),
)

for i, (xL, xR, R) in enumerate(zip(lefts, rights, radii), start=1):
    x0_vis = xL + visual_gap / 2
    x1_vis = xR - visual_gap / 2
    if x1_vis <= x0_vis:
        x0_vis, x1_vis = xL, xR

    Xv, Yv, Zv, I, J, K = cylinder_mesh_x(
        x0_vis,
        x1_vis,
        float(R),
        axis_y=axis_c,
        n_theta=n_theta,
    )

    disk_color = colors["disk_colors"][(i - 1) % len(colors["disk_colors"])]

    fig3d.add_trace(
        go.Mesh3d(
            x=Xv,
            y=Yv,
            z=Zv,
            i=I,
            j=J,
            k=K,
            color=disk_color,
            opacity=1.0,
            flatshading=False,
            lighting=dict(
                ambient=0.52,
                diffuse=0.78,
                specular=0.26,
                roughness=0.58,
                fresnel=0.035,
            ),
            lightposition=light_pos,
            hoverinfo="skip",
            showlegend=False,
            name=f"Disk {i}",
        )
    )
    disk_centers_vis.append((x0_vis + x1_vis) / 2)

if show_rims:
    rim_x_positions = []
    rim_radii = []
    for xL, xR, R in zip(lefts, rights, radii):
        x0_vis = xL + visual_gap / 2
        x1_vis = xR - visual_gap / 2
        if x1_vis <= x0_vis:
            x0_vis, x1_vis = xL, xR
        rim_x_positions.extend([x0_vis, x1_vis])
        rim_radii.extend([float(R), float(R)])

    RX, RY, RZ = rim_polyline(
        rim_x_positions,
        rim_radii,
        axis_y=axis_c,
        n_theta=max(56, n_theta),
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=RX,
            y=RY,
            z=RZ,
            mode="lines",
            line=dict(width=3, color=colors["rim"]),
            hoverinfo="skip",
            showlegend=False,
        )
    )

# Axis of rotation
axis_pad = 0.04 * (b - a)
fig3d.add_trace(
    go.Scatter3d(
        x=[a - axis_pad, b + axis_pad],
        y=[axis_c, axis_c],
        z=[0, 0],
        mode="lines",
        line=dict(width=5, color=colors["axis3d"]),
        hoverinfo="skip",
        showlegend=False,
    )
)

# Invisible hover targets at disk centers
hover_text = [
    f"Disk {i+1}<br>Interval [{lefts[i]:.5g}, {rights[i]:.5g}]"
    f"<br>x* = {x_star[i]:.5g}"
    f"<br>R = {radii[i]:.5g}"
    f"<br>ΔV = {disk_volumes[i]:.5g}"
    for i in range(n_disks)
]
fig3d.add_trace(
    go.Scatter3d(
        x=disk_centers_vis,
        y=np.full(n_disks, axis_c),
        z=np.zeros(n_disks),
        mode="markers",
        marker=dict(size=7, color="rgba(255,255,255,0.001)"),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
)

# Keep students' chosen camera angle when the slider changes.
fig3d.update_layout(
    height=820,
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False,
    paper_bgcolor=colors["paper_bg"],
    font=dict(color=colors["font"]),
    uirevision="disk-method-3d-camera",
    scene=dict(
        xaxis_title="x",
        yaxis_title="radial direction",
        zaxis_title="z",
        aspectmode="data",
        bgcolor=colors["scene_bg"],
        camera=dict(eye=dict(x=1.65, y=1.35, z=0.90)),
        xaxis=dict(
            showbackground=True,
            backgroundcolor="#F8FAFC",
            gridcolor=colors["grid"],
            zerolinecolor="#94A3B8",
            color=colors["font"],
            showspikes=False,
        ),
        yaxis=dict(
            showbackground=True,
            backgroundcolor="#F8FAFC",
            gridcolor=colors["grid"],
            zerolinecolor="#94A3B8",
            color=colors["font"],
            showspikes=False,
        ),
        zaxis=dict(
            showbackground=True,
            backgroundcolor="#F8FAFC",
            gridcolor=colors["grid"],
            zerolinecolor="#94A3B8",
            color=colors["font"],
            showspikes=False,
        ),
    ),
)

st.plotly_chart(
    fig3d,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
    },
)

if visual_gap_pct > 0:
    st.caption(
        f"The {visual_gap_pct}% gaps are only for visibility. "
        f"Each disk still represents the full mathematical thickness Δx = {dx:.6g}."
    )


# ============================================================
# Optional teacher/student data view
# ============================================================
with st.expander("Optional: disk-by-disk data"):
    rows = []
    cumulative = 0.0
    for i in range(n_disks):
        cumulative += disk_volumes[i]
        rows.append(
            {
                "Disk": i + 1,
                "left": lefts[i],
                "right": rights[i],
                "x*": x_star[i],
                "radius R": radii[i],
                "πR²Δx": disk_volumes[i],
                "cumulative volume": cumulative,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
