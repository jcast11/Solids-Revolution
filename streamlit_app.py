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
    page_title="Disk Method Lab",
    page_icon="🟠",
    layout="wide",
)

# A little visual polish without changing Streamlit's overall behavior.
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.45rem;}
        .small-note {opacity: 0.72; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


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
    """Convert scalar lambdify results (e.g. f(x)=3) to arrays."""
    if np.isscalar(values):
        return np.full_like(template, float(values), dtype=float)
    return np.asarray(values, dtype=float)


# ============================================================
# Geometry helpers
# ============================================================
def cylinder_mesh_x(x0, x1, radius, axis_y=0.0, n_theta=48):
    """Closed cylinder whose axis is parallel to x.

    Returns vertices and triangular faces for a Plotly Mesh3d trace.
    The cylinder represents ONE disk in the Riemann-sum approximation.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    y_ring = axis_y + radius * np.cos(theta)
    z_ring = radius * np.sin(theta)

    # Front ring, back ring, then two cap centers.
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

        # Side wall: two triangles per quad.
        I.extend([k, k])
        J.extend([kn, n_theta + kn])
        K.extend([n_theta + kn, n_theta + k])

        # Front cap.
        I.append(front_center)
        J.append(kn)
        K.append(k)

        # Back cap.
        I.append(back_center)
        J.append(n_theta + k)
        K.append(n_theta + kn)

    return x, y, z, I, J, K


def exact_surface_mesh(xs, radii, axis_y=0.0, n_theta=72):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X = np.tile(xs, (theta.size, 1))
    TH = np.tile(theta.reshape(-1, 1), (1, xs.size))
    RR = np.tile(radii, (theta.size, 1))
    Y = axis_y + RR * np.cos(TH)
    Z = RR * np.sin(TH)
    return X, Y, Z


def rim_polyline(x_positions, radii, axis_y=0.0, n_theta=60):
    """One Scatter3d polyline containing many circular rims."""
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
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Disk Method Controls")
    st.caption("Revolve the region between y = f(x) and y = c about the horizontal line y = c.")

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
    sample_rule = st.radio(
        "Radius sampled at",
        ["Midpoint", "Left endpoint", "Right endpoint"],
        horizontal=False,
    )

    st.divider()
    st.subheader("3D appearance")
    visual_gap_pct = st.slider(
        "Visual separation between disks",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help="This only separates the disks visually. The volume calculation still uses the full Δx.",
    )
    n_theta = st.slider("Disk roundness", 24, 96, 48, 8)
    show_rims = st.checkbox("Emphasize disk edges", value=True)
    show_true_surface = st.checkbox("Show true solid as a transparent ghost", value=True)
    show_partitions = st.checkbox("Show partition lines in the 2D region", value=True)


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

# Dense grid for curve / true solid.
xs = np.linspace(a, b, 650)
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

# Reliable numeric volume of the actual solid.
def integrand(t):
    y_val = float(np.asarray(f_num(t)))
    return np.pi * (y_val - axis_c) ** 2

try:
    exact_volume_num, quad_err = quad(integrand, a, b, limit=300)
except Exception as exc:
    st.error(f"The volume integral failed numerically. Details: {exc}")
    st.stop()

# Best-effort symbolic exact volume.
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
# Header + metric dashboard
# ============================================================
st.title("Disk Method Lab")
st.markdown(
    "Move the **number of disks** slider and watch the discrete solid converge toward the true solid of revolution. "
    "Each gold cylinder is one term of the Riemann sum."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Δx", f"{dx:.5g}")
m2.metric("Disk approximation", f"{approx_volume:.7g}")
m3.metric("Actual volume", f"{exact_volume_num:.7g}")
m4.metric(
    "Relative error",
    "—" if np.isnan(rel_error) else f"{100 * rel_error:.4g}%",
)

# ============================================================
# Main plots
# ============================================================
left_col, right_col = st.columns([0.85, 1.35], gap="large")

with left_col:
    st.subheader("1. Region and partition")
    fig2d = go.Figure()

    # Filled region between f(x) and the axis of rotation.
    fig2d.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([f_vals, np.full_like(xs, axis_c)[::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(74, 144, 226, 0.18)",
            name="Region",
            hoverinfo="skip",
        )
    )
    fig2d.add_trace(
        go.Scatter(
            x=xs,
            y=f_vals,
            mode="lines",
            line=dict(width=3, color="#3182CE"),
            name="y = f(x)",
        )
    )
    fig2d.add_trace(
        go.Scatter(
            x=[a, b],
            y=[axis_c, axis_c],
            mode="lines",
            line=dict(width=2, dash="dash", color="#374151"),
            name="axis y = c",
        )
    )

    # Sample points show exactly where each disk radius comes from.
    fig2d.add_trace(
        go.Scatter(
            x=x_star,
            y=f_star,
            mode="markers",
            marker=dict(size=7, color="#D97706"),
            name="radius samples",
            customdata=np.column_stack([np.arange(1, n_disks + 1), radii]),
            hovertemplate="Disk %{customdata[0]:.0f}<br>x*=%{x:.5g}<br>R=%{customdata[1]:.5g}<extra></extra>",
        )
    )

    if show_partitions:
        px, py = [], []
        # Draw each partition line from the axis to the curve at that x.
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
                line=dict(width=1, color="rgba(217,119,6,0.55)"),
                name="partitions",
                hoverinfo="skip",
            )
        )

    fig2d.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="x",
        yaxis_title="y",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig2d, use_container_width=True, config={"displaylogo": False})

    st.subheader("2. The Riemann sum")
    st.latex(r"\Delta x=\frac{b-a}{n}")
    st.latex(r"V_n=\sum_{i=1}^{n}\pi\,[f(x_i^*)-c]^2\,\Delta x")
    st.caption(f"Sampling rule: {sample_rule}.  Here n = {n_disks} and Δx = {dx:.6g}.")

    if sym_volume is not None:
        st.markdown("**Exact integral:**")
        st.latex(r"V=\pi\int_a^b [f(x)-c]^2\,dx=" + sp.latex(sym_volume))
    else:
        st.markdown("**Actual volume (numeric integral):**")
        st.latex(r"V=\pi\int_a^b [f(x)-c]^2\,dx")
        st.caption(f"Numerical integration error estimate: ±{quad_err:.2g}")

with right_col:
    st.subheader(f"3. Approximation by {n_disks} disks")
    fig3d = go.Figure()

    # Optional ghost surface: the smooth solid students are approximating.
    if show_true_surface:
        Xg, Yg, Zg = exact_surface_mesh(xs, true_radii, axis_y=axis_c, n_theta=max(48, n_theta))
        fig3d.add_trace(
            go.Surface(
                x=Xg,
                y=Yg,
                z=Zg,
                surfacecolor=np.zeros_like(Xg),
                colorscale=[[0, "#60A5FA"], [1, "#60A5FA"]],
                showscale=False,
                opacity=0.12,
                hoverinfo="skip",
                name="True solid",
            )
        )

    # Render each Riemann-sum disk as an ACTUAL finite-thickness cylinder.
    visual_gap = (visual_gap_pct / 100.0) * dx
    disk_centers_vis = []
    disk_radii_vis = []

    for i, (xL, xR, R) in enumerate(zip(lefts, rights, radii), start=1):
        # Gaps are purely visual; the Riemann sum still uses full dx.
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

        fig3d.add_trace(
            go.Mesh3d(
                x=Xv,
                y=Yv,
                z=Zv,
                i=I,
                j=J,
                k=K,
                color="#D9A441",
                opacity=0.96,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.45, roughness=0.5, fresnel=0.08),
                lightposition=dict(x=80, y=120, z=100),
                hoverinfo="skip",
                showlegend=False,
                name=f"Disk {i}",
            )
        )
        disk_centers_vis.append((x0_vis + x1_vis) / 2)
        disk_radii_vis.append(float(R))

    # One trace for crisp circular edges, instead of 2*n separate traces.
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

        RX, RY, RZ = rim_polyline(rim_x_positions, rim_radii, axis_y=axis_c, n_theta=max(36, n_theta))
        fig3d.add_trace(
            go.Scatter3d(
                x=RX,
                y=RY,
                z=RZ,
                mode="lines",
                line=dict(width=2, color="rgba(91, 65, 24, 0.72)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Axis of revolution.
    max_r = float(max(np.max(true_radii), np.max(radii), 1e-9))
    axis_pad = 0.04 * (b - a)
    fig3d.add_trace(
        go.Scatter3d(
            x=[a - axis_pad, b + axis_pad],
            y=[axis_c, axis_c],
            z=[0, 0],
            mode="lines",
            line=dict(width=4, color="#111827"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Invisible-ish center markers with useful hover data for each disk.
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
            marker=dict(size=5, color="rgba(0,0,0,0.01)"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    # A camera angle intentionally similar to textbook "stacked disks" figures.
    fig3d.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        scene=dict(
            xaxis_title="x",
            yaxis_title="radial y",
            zaxis_title="z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.65, y=1.35, z=0.82)),
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
    )
    st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False})

    if visual_gap_pct > 0:
        st.caption(
            f"The {visual_gap_pct}% gaps are only for visibility. Each disk still represents the full mathematical thickness Δx = {dx:.6g}."
        )
    if show_true_surface:
        st.caption("The transparent blue surface is the true solid; the gold cylinders are the disk approximation.")


# ============================================================
# Disk-by-disk data
# ============================================================
st.divider()
with st.expander("Inspect every disk", expanded=False):
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

st.markdown(
    """
    <div class="small-note">
    Teaching idea: start with n = 4 or 6 and a visible gap, then increase n while students watch the stair-stepped cylinder model converge to the transparent true solid.
    </div>
    """,
    unsafe_allow_html=True,
)