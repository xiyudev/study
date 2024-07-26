from __future__ import annotations

from manimlib import *
# from manim_imports_ext import *
from matplotlib import colormaps


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Vect3

spectral_cmap = colormaps.get_cmap("Spectral")

# Helper functions


def get_spectral_color(alpha):
    return Color(rgb=spectral_cmap(alpha)[:3])


def get_spectral_colors(n_colors, lower_bound=0, upper_bound=1):
    return [
        get_spectral_color(alpha)
        for alpha in np.linspace(lower_bound, upper_bound, n_colors)
    ]


def get_axes_and_plane(
    x_range=(0, 24),
    y_range=(-1, 1),
    z_range=(-1, 1),
    x_unit=1,
    y_unit=2,
    z_unit=2,
    origin_point=5 * LEFT,
    axes_opacity=0.5,
    plane_line_style=dict(
        stroke_color=GREY_C,
        stroke_width=1,
        stroke_opacity=0.5
    ),
):
    axes = ThreeDAxes(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        width=x_unit * (x_range[1] - x_range[0]),
        height=y_unit * (y_range[1] - y_range[0]),
        depth=z_unit * (z_range[1] - z_range[0]),
    )
    axes.shift(origin_point - axes.get_origin())
    axes.set_opacity(axes_opacity)
    axes.set_flat_stroke(False)
    plane = NumberPlane(
        axes.x_range, axes.y_range,
        width=axes.x_axis.get_length(),
        height=axes.y_axis.get_length(),
        background_line_style=plane_line_style,
        axis_config=dict(stroke_width=0),
    )
    plane.shift(axes.get_origin() - plane.get_origin())
    plane.set_flat_stroke(False)

    return axes, plane


def get_twist(wave_length, distance):
    # 350 is arbitrary. Change
    return distance / (wave_length / 350)**2


def acceleration_from_position(pos_func, time, dt=1e-3):
    p0 = pos_func(time - dt)
    p1 = pos_func(time)
    p2 = pos_func(time + dt)
    return (p0 + p2 - 2 * p1) / dt**2


def points_to_particle_info(particle, points, radius=None, c=2.0):
    """
    Given an origin, a set of points, and a radius, this returns:

    1) The unit vectors directed from the origin to each point

    2) The distances from the origin to each point

    3) An adjusted version of those distances where points
    within a given radius of the origin are considered to
    be farther away, approaching infinity at the origin.
    The intent is that when this is used for coulomb/lorenz
    forces, field vectors within a radius of a particle don't
    blow up
    """
    if radius is None:
        radius = particle.get_radius()

    if particle.track_position_history:
        approx_delays = np.linalg.norm(points - particle.get_center(), axis=1) / c
        centers = particle.get_past_position(approx_delays)
    else:
        centers = particle.get_center()

    diffs = points - centers
    norms = np.linalg.norm(diffs, axis=1)[:, np.newaxis]
    unit_diffs = np.zeros_like(diffs)
    np.true_divide(diffs, norms, out=unit_diffs, where=(norms > 0))

    adjusted_norms = norms.copy()
    mask = (0 < norms) & (norms < radius)
    adjusted_norms[mask] = radius * radius / norms[mask]
    adjusted_norms[norms == 0] = np.inf

    return unit_diffs, norms, adjusted_norms


def coulomb_force(points, particle, radius=None):
    unit_diffs, norms, adjusted_norms = points_to_particle_info(particle, points, radius)
    return particle.get_charge() * unit_diffs / adjusted_norms**2


def lorentz_force(
    points,
    particle,
    radius=None,
    c=2.0,
    epsilon0=0.025,
):
    unit_diffs, norms, adjusted_norms = points_to_particle_info(particle, points, radius, c)
    delays = norms[:, 0] / c

    acceleration = particle.get_past_acceleration(delays)
    dot_prods = (unit_diffs * acceleration).sum(1)[:, np.newaxis]
    a_perp = acceleration - dot_prods * unit_diffs

    denom = 4 * PI * epsilon0 * c**2 * adjusted_norms
    return -particle.get_charge() * a_perp / denom


# For the cylinder


class OscillatingWave(VMobject):
    def __init__(
        self,
        axes,
        y_amplitude=0.0,
        z_amplitude=0.75,
        z_phase=0.0,
        y_phase=0.0,
        wave_len=0.5,
        twist_rate=0.0,  # In rotations per unit distance
        speed=1.0,
        sample_resolution=0.005,
        stroke_width=2,
        offset=ORIGIN,
        color=None,
        **kwargs,
    ):
        self.axes = axes
        self.y_amplitude = y_amplitude
        self.z_amplitude = z_amplitude
        self.z_phase = z_phase
        self.y_phase = y_phase
        self.wave_len = wave_len
        self.twist_rate = twist_rate
        self.speed = speed
        self.sample_resolution = sample_resolution
        self.offset = offset

        super().__init__(**kwargs)

        color = color or self.get_default_color(wave_len)
        self.set_stroke(color, stroke_width)
        self.set_flat_stroke(False)

        self.time = 0
        self.clock_is_stopped = False

        self.add_updater(lambda m, dt: m.update_points(dt))

    def update_points(self, dt):
        if not self.clock_is_stopped:
            self.time += dt
        xs = np.arange(
            self.axes.x_axis.x_min,
            self.axes.x_axis.x_max,
            self.sample_resolution
        )
        self.set_points_as_corners(
            self.offset + self.xt_to_point(xs, self.time)
        )

    def stop_clock(self):
        self.clock_is_stopped = True

    def start_clock(self):
        self.clock_is_stopped = False

    def toggle_clock(self):
        self.clock_is_stopped = not self.clock_is_stopped

    def xt_to_yz(self, x, t):
        phase = TAU * t * self.speed / self.wave_len
        y_outs = self.y_amplitude * np.sin(TAU * x / self.wave_len - phase - self.y_phase)
        z_outs = self.z_amplitude * np.sin(TAU * x / self.wave_len - phase - self.z_phase)
        twist_angles = x * self.twist_rate * TAU
        y = np.cos(twist_angles) * y_outs - np.sin(twist_angles) * z_outs
        z = np.sin(twist_angles) * y_outs + np.cos(twist_angles) * z_outs

        return y, z

    def xt_to_point(self, x, t):
        y, z = self.xt_to_yz(x, t)
        return self.axes.c2p(x, y, z)

    def get_default_color(self, wave_len):
        return get_spectral_color(inverse_interpolate(
            1.5, 0.5, wave_len
        ))


class MeanWave(VMobject):
    def __init__(self, waves, **kwargs):
        self.waves = waves
        self.offset = np.array(ORIGIN)
        self.time = 0
        super().__init__(**kwargs)
        self.set_flat_stroke(False)
        self.add_updater(lambda m, dt: m.update_points(dt))

    def update_points(self, dt):
        for wave in self.waves:
            wave.update_points(dt)

        self.time += dt

        points = sum(wave.get_points() for wave in self.waves) / len(self.waves)
        self.set_points(points)

    def xt_to_yz(self, x, t):
        return tuple(
            np.array([
                wave.xt_to_yz(x, t)[i]
                for wave in self.waves
            ]).mean(0)
            for i in (0, 1)
        )


class SugarCylinder(Cylinder):
    def __init__(
        self, axes, camera,
        radius=0.5,
        color=BLUE_A,
        opacity=0.2,
        shading=(0.5, 0.5, 0.5),
        resolution=(51, 101),
    ):
        super().__init__(
            color=color,
            opacity=opacity,
            resolution=resolution,
            shading=shading,
        )
        self.set_width(2 * axes.z_axis.get_unit_size() * radius)
        self.set_depth(axes.x_axis.get_length(), stretch=True)
        self.rotate(PI / 2, UP)
        self.move_to(axes.get_origin(), LEFT)
        # self.set_shading(*shading)
        self.always_sort_to_camera(camera)


class Polarizer(VGroup):
    def __init__(
        self, axes,
        radius=1.0,
        angle=0,
        stroke_color=GREY_C,
        stroke_width=2,
        fill_color=GREY_C,
        fill_opacity=0.25,
        n_lines=14,
        line_opacity=0.2,
        arrow_stroke_color=WHITE,
        arrow_stroke_width=5,

    ):
        true_radius = radius * axes.z_axis.get_unit_size()
        circle = Circle(
            radius=true_radius,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
        )

        lines = VGroup(*(
            Line(circle.pfp(a), circle.pfp(1 - a))
            for a in np.arccos(np.linspace(1, -1, n_lines + 2)[1:-1]) / TAU
        ))
        lines.set_stroke(WHITE, 1, opacity=line_opacity)

        arrow = Vector(
            0.5 * true_radius * UP,
            stroke_color=arrow_stroke_color,
            stroke_width=arrow_stroke_width,
        )
        arrow.move_to(circle.get_top(), DOWN)

        super().__init__(
            circle, lines, arrow,
            # So the center works correctly
            VectorizedPoint(circle.get_bottom() + arrow.get_height() * DOWN),
        )
        self.set_flat_stroke(True)
        self.rotate(PI / 2, RIGHT)
        self.rotate(PI / 2, IN)
        self.rotate(angle, RIGHT)
        self.rotate(1 * DEGREES, UP)


class ProbagatingRings(VGroup):
    def __init__(
        self, line,
        n_rings=5,
        start_width=3,
        width_decay_rate=0.1,
        stroke_color=WHITE,
        growth_rate=2.0,
        spacing=0.2,
    ):
        ring = Circle(radius=1e-3, n_components=101)
        ring.set_stroke(stroke_color, start_width)
        ring.apply_matrix(z_to_vector(line.get_vector()))
        ring.move_to(line)
        ring.set_flat_stroke(False)

        super().__init__(*ring.replicate(n_rings))

        self.growth_rate = growth_rate
        self.spacing = spacing
        self.width_decay_rate = width_decay_rate
        self.start_width = start_width
        self.time = 0

        self.add_updater(lambda m, dt: self.update_rings(dt))

    def update_rings(self, dt):
        if dt == 0:
            return
        self.time += dt
        space = 0
        for ring in self.submobjects:
            effective_time = max(self.time - space, 0)
            target_radius = max(effective_time * self.growth_rate, 1e-3)
            ring.scale(target_radius / ring.get_radius())
            space += self.spacing
            ring.set_stroke(width=np.exp(-self.width_decay_rate * effective_time))
        return self


class TwistedRibbon(ParametricSurface):
    def __init__(
        self,
        axes,
        amplitude,
        twist_rate,
        start_point=(0, 0, 0),
        color=WHITE,
        opacity=0.4,
        resolution=(101, 11),
    ):
        super().__init__(
            lambda u, v: axes.c2p(
                u,
                v * amplitude * np.sin(TAU * twist_rate * u),
                v * amplitude * np.cos(TAU * twist_rate * u)
            ),
            u_range=axes.x_range[:2],
            v_range=(-1, 1),
            color=color,
            opacity=opacity,
            resolution=resolution,
            prefered_creation_axis=0,
        )
        self.shift(axes.c2p(*start_point) - axes.get_origin())


# For fields


class ChargedParticle(Group):
    def __init__(
        self,
        point=ORIGIN,
        charge=1.0,
        mass=1.0,
        color=RED,
        show_sign=True,
        sign="+",
        radius=0.2,
        rotation=0,
        sign_stroke_width=2,
        track_position_history=True,
        history_size=7200,
        euler_steps_per_frame=10,
    ):
        self.charge = charge
        self.mass = mass

        sphere = TrueDot(radius=radius, color=color)
        sphere.make_3d()
        sphere.move_to(point)
        self.sphere = sphere

        self.track_position_history = track_position_history
        self.history_size = history_size
        self.velocity = np.zeros(3)  # Only used if force are added
        self.euler_steps_per_frame = euler_steps_per_frame
        self.init_clock(point)

        super().__init__(sphere)

        if show_sign:
            sign = Tex(sign)
            sign.set_width(radius)
            sign.rotate(rotation, RIGHT)
            sign.set_stroke(WHITE, sign_stroke_width)
            sign.move_to(sphere)
            self.add(sign)
            self.sign = sign

    # Related to updaters

    def update(self, dt: float = 0, recurse: bool = True):
        super().update(dt, recurse)
        # Do this instead of adding an updater, because
        # otherwise all animations require the
        # suspend_mobject_updating=false flag
        self.increment_clock(dt)

    def init_clock(self, start_point):
        self.time = 0
        self.time_step = 1 / 30  # This will be updated
        self.recent_positions = np.tile(start_point, 3).reshape((3, 3))
        if self.track_position_history:
            self.position_history = np.zeros((self.history_size, 3))
            self.acceleration_history = np.zeros((self.history_size, 3))
            self.history_index = -1

    def increment_clock(self, dt):
        if dt == 0:
            return self
        self.time += dt
        self.time_step = dt
        self.recent_positions[0:2] = self.recent_positions[1:3]
        self.recent_positions[2] = self.get_center()
        if self.track_position_history:
            self.add_to_position_history()

    def add_to_position_history(self):
        self.history_index += 1
        hist_size = self.history_size
        # If overflowing, copy second half of history
        # lists to the first half, and reset index
        if self.history_index >= hist_size:
            for arr in [self.position_history, self.acceleration_history]:
                arr[:hist_size // 2, :] = arr[hist_size // 2:, :]
            self.history_index = (hist_size // 2) + 1

        self.position_history[self.history_index] = self.get_center()
        self.acceleration_history[self.history_index] = self.get_acceleration()
        return self

    def ignore_last_motion(self):
        self.recent_positions[:] = self.get_center()
        return self

    def add_force(self, force_func: Callable[[Vect3], Vect3]):
        espf = self.euler_steps_per_frame

        def update_from_force(particle, dt):
            if dt == 0:
                return
            for _ in range(espf):
                acc = force_func(particle.get_center()) / self.mass
                self.velocity += acc * dt / espf
                self.shift(self.velocity * dt / espf)

        self.add_updater(update_from_force)
        return self

    def add_spring_force(self, k=1.0, center=None):
        center = center if center is not None else self.get_center().copy()
        self.add_force(lambda p: k * (center - p))
        return self

    def add_field_force(self, field):
        charge = self.get_charge()
        self.add_force(lambda p: charge * field.get_forces([p])[0])
        return self

    def fix_x(self):
        x = self.get_x()
        self.add_updater(lambda m: m.set_x(x))

    # Getters

    def get_charge(self):
        return self.charge

    def get_radius(self):
        return self.sphere.get_radius()

    def get_internal_time(self):
        return self.time

    def scale(self, factor, *args, **kwargs):
        super().scale(factor, *args, **kwargs)
        self.sphere.set_radius(factor * self.sphere.get_radius())
        return self

    def get_acceleration(self):
        p0, p1, p2 = self.recent_positions
        # if (p0 == p1).all() or (p1 == p2).all():
        if np.isclose(p0, p1).all() or np.isclose(p1, p2).all():
            # Otherwise, starts and stops have artificially
            # high acceleration
            return np.zeros(3)
        return (p0 + p2 - 2 * p1) / self.time_step**2

    def get_info_from_delays(self, info_arr, delays):
        if not hasattr(self, "acceleration_history"):
            raise Exception("track_position_history is not turned on")

        if len(info_arr) == 0:
            return np.zeros((len(delays), 3))

        pre_indices = self.history_index - delays / self.time_step
        indices = np.clip(pre_indices, 0, self.history_index).astype(int)

        return info_arr[indices]

    def get_past_acceleration(self, delays):
        return self.get_info_from_delays(self.acceleration_history, delays)

    def get_past_position(self, delays):
        return self.get_info_from_delays(self.position_history, delays)


class AccelerationVector(Vector):
    def __init__(
        self,
        particle,
        stroke_color=PINK,
        stroke_width=4,
        flat_stroke=False,
        norm_func=lambda n: np.tanh(n),
        **kwargs
    ):
        self.norm_func = norm_func

        super().__init__(
            RIGHT,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            flat_stroke=flat_stroke,
            **kwargs
        )
        self.add_updater(lambda m: m.pin_to_particle(particle))

    def pin_to_particle(self, particle):
        a_vect = particle.get_acceleration()
        norm = get_norm(a_vect)
        if self.norm_func is not None and norm > 0:
            a_vect *= self.norm_func(norm) / norm
        center = particle.get_center()
        self.put_start_and_end_on(center, center + a_vect)


class VectorField(VMobject):
    def __init__(
        self,
        func,
        stroke_color=BLUE,
        center=ORIGIN,
        x_density=2.0,
        y_density=2.0,
        z_density=2.0,
        width=14,
        height=8,
        depth=0,
        stroke_width: float = 2,
        tip_width_ratio: float = 4,
        tip_len_to_width: float = 0.01,
        max_vect_len: float | None = None,
        min_drawn_norm: float = 1e-2,
        flat_stroke=False,
        norm_to_opacity_func=None,
        norm_to_rgb_func=None,
        **kwargs
    ):
        self.func = func
        self.stroke_width = stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.min_drawn_norm = min_drawn_norm
        self.norm_to_opacity_func = norm_to_opacity_func
        self.norm_to_rgb_func = norm_to_rgb_func

        if max_vect_len is not None:
            self.max_vect_len = max_vect_len
        else:
            densities = np.array([x_density, y_density, z_density])
            dims = np.array([width, height, depth])
            self.max_vect_len = 1.0 / densities[dims > 0].mean()

        self.sample_points = self.get_sample_points(
            center, width, height, depth,
            x_density, y_density, z_density
        )
        self.init_base_stroke_width_array(len(self.sample_points))

        super().__init__(
            stroke_color=stroke_color,
            flat_stroke=flat_stroke,
            **kwargs
        )

        n_samples = len(self.sample_points)
        self.set_points(np.zeros((8 * n_samples - 1, 3)))
        self.set_stroke(width=stroke_width)
        self.update_vectors()

    def get_sample_points(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        depth: float,
        x_density: float,
        y_density: float,
        z_density: float
    ) -> np.ndarray:
        to_corner = np.array([width / 2, height / 2, depth / 2])
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        to_corner = spacings * (to_corner / spacings).astype(int)
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        return cartesian_product(*(
            np.arange(low, high, space)
            for low, high, space in zip(lower_corner, upper_corner, spacings)
        ))

    def init_base_stroke_width_array(self, n_sample_points):
        arr = np.ones(8 * n_sample_points - 1)
        arr[4::8] = self.tip_width_ratio
        arr[5::8] = self.tip_width_ratio * 0.5
        arr[6::8] = 0
        arr[7::8] = 0
        self.base_stroke_width_array = arr

    def set_stroke(self, color=None, width=None, opacity=None, background=None, recurse=True):
        super().set_stroke(color, None, opacity, background, recurse)
        if width is not None:
            self.set_stroke_width(float(width))
        return self

    def set_stroke_width(self, width: float):
        if self.get_num_points() > 0:
            self.get_stroke_widths()[:] = width * self.base_stroke_width_array
            self.stroke_width = width
        return self

    def update_vectors(self):
        tip_width = self.tip_width_ratio * self.stroke_width
        tip_len = self.tip_len_to_width * tip_width
        samples = self.sample_points

        # Get raw outputs and lengths
        outputs = self.func(samples)
        norms = np.linalg.norm(outputs, axis=1)[:, np.newaxis]

        # How long should the arrows be drawn?
        max_len = self.max_vect_len
        if max_len < np.inf:
            drawn_norms = max_len * np.tanh(norms / max_len)
        else:
            drawn_norms = norms

        # What's the distance from the base of an arrow to
        # the base of its head?
        dist_to_head_base = np.clip(drawn_norms - tip_len, 0, np.inf)

        # Set all points
        unit_outputs = np.zeros_like(outputs)
        np.true_divide(outputs, norms, out=unit_outputs, where=(norms > self.min_drawn_norm))

        points = self.get_points()
        points[0::8] = samples
        points[2::8] = samples + dist_to_head_base * unit_outputs
        points[4::8] = points[2::8]
        points[6::8] = samples + drawn_norms * unit_outputs
        for i in (1, 3, 5):
            points[i::8] = 0.5 * (points[i - 1::8] + points[i + 1::8])
        points[7::8] = points[6:-1:8]

        # Adjust stroke widths
        width_arr = self.stroke_width * self.base_stroke_width_array
        width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        width_scalars = np.repeat(width_scalars, 8)[:-1]
        self.get_stroke_widths()[:] = width_scalars * width_arr

        # Potentially adjust opacity and color
        if self.norm_to_opacity_func is not None:
            self.get_stroke_opacities()[:] = self.norm_to_opacity_func(
                np.repeat(norms, 8)[:-1]
            )
        if self.norm_to_rgb_func is not None:
            self.get_stroke_colors()
            self.data['stroke_rgba'][:, :3] = self.norm_to_rgb_func(
                np.repeat(norms, 8)[:-1]
            )

        self.note_changed_data()
        return self


class TimeVaryingVectorField(VectorField):
    def __init__(
        self,
        # Takes in an array of points and a float for time
        time_func,
        **kwargs
    ):
        self.time = 0
        super().__init__(func=lambda p: time_func(p, self.time), **kwargs)
        self.add_updater(lambda m, dt: m.increment_time(dt))
        always(self.update_vectors)

    def increment_time(self, dt):
        self.time += dt


class ChargeBasedVectorField(VectorField):
    default_color = BLUE

    def __init__(self, *charges, **kwargs):
        self.charges = list(charges)
        super().__init__(
            self.get_forces,
            color=kwargs.pop("color", self.default_color),
            **kwargs
        )
        self.add_updater(lambda m: m.update_vectors())

    def get_forces(self, points):
        # To be implemented in subclasses
        return np.zeros_like(points)


class CoulombField(ChargeBasedVectorField):
    default_color = YELLOW

    def get_forces(self, points):
        return sum(
            coulomb_force(points, charge)
            for charge in self.charges
        )


class LorentzField(ChargeBasedVectorField):
    def __init__(
        self, *charges,
        radius_of_suppression=None,
        c=2.0,
        **kwargs
    ):
        self.radius_of_suppression = radius_of_suppression
        self.c = c
        super().__init__(*charges, **kwargs)

    def get_forces(self, points):
        return sum(
            lorentz_force(
                points, charge,
                radius=self.radius_of_suppression,
                c=self.c
            )
            for charge in self.charges
        )


class ColoumbPlusLorentzField(LorentzField):
    def get_forces(self, points):
        return sum(
            lorentz_force(
                points, charge,
                radius=self.radius_of_suppression,
                c=self.c
            ) + sum(
                coulomb_force(points, charge)
                for charge in self.charges
            )
            for charge in self.charges
        )


class GraphAsVectorField(VectorField):
    def __init__(
        self,
        axes: Axes | ThreeDAxes,
        # Maps x to y, or x to (y, z)
        graph_func: Callable[[VectN], VectN] | Callable[[VectN], Tuple[VectN, VectN]],
        x_density=10.0,
        max_vect_len=np.inf,
        **kwargs,
    ):
        self.sample_xs = np.arange(axes.x_axis.x_min, axes.x_axis.x_max, 1.0 / x_density)
        self.axes = axes

        def vector_func(points):
            output = graph_func(self.sample_xs)
            if isinstance(axes, ThreeDAxes):
                graph_points = axes.c2p(self.sample_xs, *output)
            else:
                graph_points = axes.c2p(self.sample_xs, output)
            base_points = axes.x_axis.n2p(self.sample_xs)
            return graph_points - base_points

        super().__init__(
            func=vector_func,
            max_vect_len=max_vect_len,
            **kwargs
        )
        always(self.update_vectors)

    def reset_sample_points(self):
        self.sample_points = self.get_sample_points()

    def get_sample_points(self, *args, **kwargs):
        # Override super class and ignore all length/density information
        return self.axes.x_axis.n2p(self.sample_xs)


class OscillatingFieldWave(GraphAsVectorField):
    def __init__(self, axes, wave, **kwargs):
        self.wave = wave
        if "stroke_color" not in kwargs:
            kwargs["stroke_color"] = wave.get_color()
        super().__init__(
            axes=axes,
            graph_func=lambda x: wave.xt_to_yz(x, wave.time),
            **kwargs
        )

    def get_sample_points(self, *args, **kwargs):
        # Override super class and ignore all length/density information
        return self.wave.offset + self.axes.x_axis.n2p(self.sample_xs)


# Structure


class Molecule(Group):
    # List of characters
    atoms = []

    # List of 3d coordinates
    coordinates = np.zeros((0, 3))

    # List of pairs of indices
    bonds = []

    atom_to_color = {
        "H": WHITE,
        "O": RED,
        "C": GREY,
    }
    atom_to_radius = {
        "H": 0.1,
        "O": 0.2,
        "C": 0.19,
    }
    ball_config = dict(shading=(0.25, 0.5, 0.5), glow_factor=0.25)
    stick_config = dict(stroke_width=1, stroke_color=GREY_A, flat_stroke=False)

    def __init__(self, height=2.0, **kwargs):
        coords = np.array(self.coordinates)
        radii = np.array([self.atom_to_radius[atom] for atom in self.atoms])
        rgbas = np.array([color_to_rgba(self.atom_to_color[atom]) for atom in self.atoms])

        balls = DotCloud(coords, **self.ball_config)
        balls.set_radii(radii)
        balls.set_rgba_array(rgbas)

        sticks = VGroup()
        for i, j in self.bonds:
            c1, c2 = coords[[i, j], :]
            r1, r2 = radii[[i, j]]
            unit_vect = normalize(c2 - c1)

            sticks.add(Line(
                c1 + r1 * unit_vect, c2 - r2 * unit_vect,
                **self.stick_config
            ))

        super().__init__(balls, sticks, **kwargs)

        self.apply_depth_test()
        self.balls = balls
        self.sticks = sticks
        self.set_height(height)


class Sucrose(Molecule):
    atoms = [
        "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
        "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
        "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H",
    ]
    coordinates = np.array([
        [-1.468 ,  0.4385, -0.9184],
        [-0.6033, -0.8919,  0.8122],
        [ 0.9285,  0.4834, -0.3053],
        [-3.0702, -2.0054,  1.1933],
        [-4.62  ,  0.6319,  0.7326],
        [ 1.2231,  0.2156,  2.5658],
        [ 3.6108, -1.7286,  0.6379],
        [ 3.15  ,  1.8347,  1.1537],
        [-1.9582, -1.848 , -2.43  ],
        [-1.3845,  3.245 , -0.8933],
        [ 3.8369,  0.2057, -2.5044],
        [-1.4947, -0.8632, -0.3037],
        [-2.9301, -1.0229,  0.1866],
        [-3.229 ,  0.3737,  0.6887],
        [-2.5505,  1.2243, -0.3791],
        [ 0.7534, -0.7453,  0.3971],
        [ 1.6462, -0.7853,  1.639 ],
        [ 3.1147, -0.5553,  1.2746],
        [ 3.2915,  0.6577,  0.3521],
        [ 2.2579,  0.7203, -0.7858],
        [-1.0903, -1.9271, -1.3122],
        [-2.0027,  2.5323,  0.1653],
        [ 2.5886, -0.1903, -1.9666],
        [-3.6217, -1.2732, -0.6273],
        [-2.8148,  0.5301,  1.6917],
        [-3.2289,  1.4361, -1.215 ],
        [ 1.0588, -1.5992, -0.2109],
        [ 1.5257, -1.753 ,  2.1409],
        [ 3.6908, -0.4029,  2.1956],
        [ 4.31  ,  0.675 , -0.0511],
        [ 2.2441,  1.7505, -1.1644],
        [-1.1311, -2.9324, -0.8803],
        [-0.0995, -1.7686, -1.74  ],
        [-1.2448,  2.3605,  0.9369],
        [-2.799 ,  3.1543,  0.5841],
        [ 1.821 , -0.1132, -2.7443],
        [ 2.6532, -1.2446, -1.6891],
        [-3.98  , -1.9485,  1.5318],
        [-4.7364,  1.5664,  0.9746],
        [ 0.2787,  0.0666,  2.7433],
        [ 4.549 , -1.5769,  0.4327],
        [ 3.3427,  2.6011,  0.5871],
        [-1.6962, -2.5508, -3.0488],
        [-0.679 ,  2.6806, -1.2535],
        [ 3.7489,  1.1234, -2.8135],
    ])
    bonds = [
        (0, 11),
        (0, 14),
        (1, 11),
        (1, 15),
        (2, 15),
        (2, 19),
        (3, 12),
        (3, 37),
        (4, 13),
        (4, 38),
        (5, 16),
        (5, 39),
        (6, 17),
        (6, 40),
        (7, 18),
        (7, 41),
        (8, 20),
        (8, 42),
        (9, 21),
        (9, 43),
        (10, 22),
        (10, 44),
        (11, 12),
        (11, 20),
        (12, 13),
        (12, 23),
        (13, 14),
        (13, 24),
        (14, 21),
        (14, 25),
        (15, 16),
        (15, 26),
        (16, 17),
        (16, 27),
        (17, 18),
        (17, 28),
        (18, 19),
        (18, 29),
        (19, 22),
        (19, 30),
        (20, 31),
        (20, 32),
        (21, 33),
        (21, 34),
        (22, 35),
        (22, 36),
    ]


class Carbonate(Molecule):
    # List of characters
    atoms = ["O", "O", "O", "C", "H", "H"]

    # List of 3d coordinates
    coordinates = np.array([
       [-6.9540e-01, -1.1061e+00,  0.0000e+00],
       [-6.9490e-01,  1.1064e+00,  0.0000e+00],
       [ 1.3055e+00, -3.0000e-04,  1.0000e-04],
       [ 8.4700e-02,  0.0000e+00, -1.0000e-04],
       [-1.6350e-01, -1.9304e+00,  1.0000e-04],
       [-1.6270e-01,  1.9305e+00,  1.0000e-04],
    ])

    # List of pairs of indices
    bonds = [(0, 3), (0, 4), (1, 3), (1, 5), (2, 3)]
    bond_types = [1, 1, 1, 1, 2]


class Calcite(Molecule):
    atoms = [
        "C", "C", "Ca", "C", "O", "O", "C", "Ca", "C", "O", "O", "O", "Ca", "C", "O", "O", "C", "Ca", "C", "O", "C", "Ca", "C", "O", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "C", "Ca", "C", "O", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "O", "C", "C", "Ca", "C", "O", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "C", "Ca", "C", "O", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "O", "Ca", "C", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "O", "Ca", "C", "O", "O", "O", "Ca", "C", "O", "Ca", "C", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "Ca", "C", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "O", "Ca", "Ca", "C", "O", "O", "O", "O", "O", "Ca", "C", "O", "O", "O", "O", "O", "O",
    ]
    atom_to_color = {
        "Ca": GREEN,
        "O": RED,
        "C": GREY,
    }
    atom_to_radius = {
        "Ca": 0.25,
        "O": 0.2,
        "C": 0.19,
    }

    coordinates = np.array([
        [-1.43769337, -2.49015797,  1.41822405],
        [-1.43769337,  2.49015797,  1.41822405],
        [-2.87538675,   .00000000,  2.83644809],
        [-2.87538675,   .00000000,  7.09112022],
        [-2.48577184,  1.88504958,  1.41822404],
        [-1.43769337, -1.27994119,  1.41822404],
        [-1.43769337,  7.47047390,  1.41822405],
        [-2.87538675,  4.98031594,  2.83644809],
        [-2.87538675,  4.98031594,  7.09112022],
        [-2.48577184,  6.86536552,  1.41822404],
        [-1.43769337,  3.70037474,  1.41822404],
        [-2.87538675,  1.21021677,  7.09112022],
        [-2.87538675,  9.96063187,  2.83644809],
        [-2.87538675,  9.96063187,  7.09112022],
        [-1.43769337,  8.68069068,  1.41822404],
        [-2.87538675,  6.19053271,  7.09112022],
        [ 2.87538675,   .00000000,  1.41822405],
        [ 1.43769337, -2.49015797,  2.83644809],
        [ 1.43769337, -2.49015797,  7.09112022],
        [ 1.82730828,  -.60510839,  1.41822404],
        [ 2.87538675,  4.98031594,  1.41822405],
        [ 1.43769337,  2.49015797,  2.83644809],
        [ 1.43769337,  2.49015797,  7.09112022],
        [ -.38961490,  1.88504958,  1.41822404],
        [ 1.82730828,  4.37520755,  1.41822404],
        [ 2.87538675,  1.21021677,  1.41822404],
        [  .00000000,   .00000000,   .00000000],
        [  .00000000,   .00000000,  8.50934427],
        [  .00000000,   .00000000,  4.25467213],
        [-1.04807847,   .60510839,  4.25467213],
        [ 1.04807847,   .60510839,  4.25467213],
        [  .00000000, -1.21021677,  4.25467213],
        [-1.82730828,  -.60510839,  7.09112022],
        [  .38961490,  1.88504958,  7.09112022],
        [ 1.43769337, -1.27994119,  7.09112022],
        [-1.43769337, -2.49015797,  5.67289618],
        [-1.43769337, -2.49015797,  9.92756831],
        [-2.48577184, -1.88504958,  9.92756831],
        [ -.38961490, -1.88504958,  9.92756831],
        [ 2.87538675,  9.96063187,  1.41822405],
        [ 1.43769337,  7.47047390,  2.83644809],
        [ 1.43769337,  7.47047390,  7.09112022],
        [ -.38961490,  6.86536552,  1.41822404],
        [ 1.82730828,  9.35552349,  1.41822404],
        [ 2.87538675,  6.19053271,  1.41822404],
        [  .00000000,  4.98031594,   .00000000],
        [  .00000000,  4.98031594,  8.50934427],
        [  .00000000,  4.98031594,  4.25467213],
        [-1.04807847,  5.58542432,  4.25467213],
        [ 1.04807847,  5.58542432,  4.25467213],
        [  .00000000,  3.77009916,  4.25467213],
        [-1.82730828,  4.37520755,  7.09112022],
        [  .38961490,  6.86536552,  7.09112022],
        [ 1.43769337,  3.70037474,  7.09112022],
        [-1.43769337,  2.49015797,  5.67289618],
        [-1.43769337,  2.49015797,  9.92756831],
        [-2.48577184,  3.09526635,  9.92756831],
        [ -.38961490,  3.09526635,  9.92756831],
        [-1.43769337,  1.27994120,  9.92756831],
        [  .00000000,  9.96063187,   .00000000],
        [  .00000000,  9.96063187,  8.50934427],
        [  .00000000,  9.96063187,  4.25467213],
        [-1.04807847, 10.56574026,  4.25467213],
        [ 1.04807847, 10.56574026,  4.25467213],
        [  .00000000,  8.75041510,  4.25467213],
        [-1.82730828,  9.35552349,  7.09112022],
        [ 1.43769337,  8.68069068,  7.09112022],
        [-1.43769337,  7.47047390,  5.67289618],
        [-1.43769337,  7.47047390,  9.92756831],
        [-2.48577184,  8.07558229,  9.92756831],
        [ -.38961490,  8.07558229,  9.92756831],
        [-1.43769337,  6.26025713,  9.92756831],
        [ 7.18846687, -2.49015797,  1.41822405],
        [ 7.18846687,  2.49015797,  1.41822405],
        [ 5.75077349,   .00000000,  2.83644809],
        [ 5.75077349,   .00000000,  7.09112022],
        [ 3.92346522,  -.60510839,  1.41822404],
        [ 6.14038840,  1.88504958,  1.41822404],
        [ 7.18846687, -1.27994119,  1.41822404],
        [ 4.31308012, -2.49015797,   .00000000],
        [ 4.31308012, -2.49015797,  8.50934427],
        [ 4.31308012, -2.49015797,  4.25467213],
        [ 3.26500165, -1.88504958,  4.25467213],
        [ 5.36115859, -1.88504958,  4.25467213],
        [ 4.70269502,  -.60510839,  7.09112022],
        [ 7.18846687,  7.47047390,  1.41822405],
        [ 5.75077349,  4.98031594,  2.83644809],
        [ 5.75077349,  4.98031594,  7.09112022],
        [ 3.92346522,  4.37520755,  1.41822404],
        [ 6.14038840,  6.86536552,  1.41822404],
        [ 7.18846687,  3.70037474,  1.41822404],
        [ 4.31308012,  2.49015797,   .00000000],
        [ 4.31308012,  2.49015797,  8.50934427],
        [ 4.31308012,  2.49015797,  4.25467213],
        [ 3.26500165,  3.09526635,  4.25467213],
        [ 5.36115859,  3.09526635,  4.25467213],
        [ 4.31308012,  1.27994120,  4.25467213],
        [ 2.48577184,  1.88504958,  7.09112022],
        [ 4.70269502,  4.37520755,  7.09112022],
        [ 5.75077349,  1.21021677,  7.09112022],
        [ 2.87538675,   .00000000,  5.67289618],
        [ 2.87538675,   .00000000,  9.92756831],
        [ 1.82730828,   .60510839,  9.92756831],
        [ 3.92346522,   .60510839,  9.92756831],
        [ 2.87538675, -1.21021677,  9.92756831],
        [ 5.75077349,  9.96063187,  2.83644809],
        [ 5.75077349,  9.96063187,  7.09112022],
        [ 3.92346522,  9.35552349,  1.41822404],
        [ 7.18846687,  8.68069068,  1.41822404],
        [ 4.31308012,  7.47047390,   .00000000],
        [ 4.31308012,  7.47047390,  8.50934427],
        [ 4.31308012,  7.47047390,  4.25467213],
        [ 3.26500165,  8.07558229,  4.25467213],
        [ 5.36115859,  8.07558229,  4.25467213],
        [ 4.31308012,  6.26025713,  4.25467213],
        [ 2.48577184,  6.86536552,  7.09112022],
        [ 4.70269502,  9.35552349,  7.09112022],
        [ 5.75077349,  6.19053271,  7.09112022],
        [ 2.87538675,  4.98031594,  5.67289618],
        [ 2.87538675,  4.98031594,  9.92756831],
        [ 1.82730828,  5.58542432,  9.92756831],
        [ 3.92346522,  5.58542432,  9.92756831],
        [ 2.87538675,  3.77009916,  9.92756831],
        [ 2.87538675,  9.96063187,  5.67289618],
        [ 2.87538675,  9.96063187,  9.92756831],
        [ 1.82730828, 10.56574026,  9.92756831],
        [ 3.92346522, 10.56574026,  9.92756831],
        [ 2.87538675,  8.75041510,  9.92756831],
        [10.06385361, -2.49015797,  2.83644809],
        [10.06385361, -2.49015797,  7.09112022],
        [10.45346852,  -.60510839,  1.41822404],
        [10.06385361,  2.49015797,  2.83644809],
        [10.06385361,  2.49015797,  7.09112022],
        [ 8.23654533,  1.88504958,  1.41822404],
        [10.45346852,  4.37520755,  1.41822404],
        [ 8.62616024,   .00000000,   .00000000],
        [ 8.62616024,   .00000000,  8.50934427],
        [ 8.62616024,   .00000000,  4.25467213],
        [ 7.57808177,   .60510839,  4.25467213],
        [ 9.67423871,   .60510839,  4.25467213],
        [ 8.62616024, -1.21021677,  4.25467213],
        [ 6.79885196,  -.60510839,  7.09112022],
        [ 9.01577514,  1.88504958,  7.09112022],
        [10.06385361, -1.27994119,  7.09112022],
        [ 7.18846687, -2.49015797,  5.67289618],
        [ 7.18846687, -2.49015797,  9.92756831],
        [ 6.14038840, -1.88504958,  9.92756831],
        [ 8.23654533, -1.88504958,  9.92756831],
        [10.06385361,  7.47047390,  2.83644809],
        [10.06385361,  7.47047390,  7.09112022],
        [ 8.23654533,  6.86536552,  1.41822404],
        [10.45346852,  9.35552349,  1.41822404],
        [ 8.62616024,  4.98031594,   .00000000],
        [ 8.62616024,  4.98031594,  8.50934427],
        [ 8.62616024,  4.98031594,  4.25467213],
        [ 7.57808177,  5.58542432,  4.25467213],
        [ 9.67423871,  5.58542432,  4.25467213],
        [ 8.62616024,  3.77009916,  4.25467213],
        [ 6.79885196,  4.37520755,  7.09112022],
        [ 9.01577514,  6.86536552,  7.09112022],
        [10.06385361,  3.70037474,  7.09112022],
        [ 7.18846687,  2.49015797,  5.67289618],
        [ 7.18846687,  2.49015797,  9.92756831],
        [ 6.14038840,  3.09526635,  9.92756831],
        [ 8.23654533,  3.09526635,  9.92756831],
        [ 7.18846687,  1.27994120,  9.92756831],
        [ 8.62616024,  9.96063187,   .00000000],
        [ 8.62616024,  9.96063187,  8.50934427],
        [ 8.62616024,  9.96063187,  4.25467213],
        [ 7.57808177, 10.56574026,  4.25467213],
        [ 9.67423871, 10.56574026,  4.25467213],
        [ 8.62616024,  8.75041510,  4.25467213],
        [ 6.79885196,  9.35552349,  7.09112022],
        [10.06385361,  8.68069068,  7.09112022],
        [ 7.18846687,  7.47047390,  5.67289618],
        [ 7.18846687,  7.47047390,  9.92756831],
        [ 6.14038840,  8.07558229,  9.92756831],
        [ 8.23654533,  8.07558229,  9.92756831],
        [ 7.18846687,  6.26025713,  9.92756831],
        [10.45346852,   .60510839,  9.92756831],
        [10.45346852,  5.58542432,  9.92756831],
        [10.45346852, 10.56574026,  9.92756831],
    ])

class ContinuousWave(InteractiveScene):
    def construct(self):
        wave = self.get_wave()
        points = wave.get_points().copy()
        wave.add_updater(lambda m: m.set_points(points).stretch(
            (1 - math.sin(self.time)), 1,
        ))
        self.add(wave)
        self.wait(20)

    def get_wave(self):
        axes = Axes((0, 2 * TAU), (0, 1))
        wave = axes.get_graph(np.sin)
        wave.set_stroke(BLUE, 4)
        wave.set_width(2.0)
        return wave


class DiscreteWave(ContinuousWave):
    def construct(self):
        # Test
        waves = self.get_wave().replicate(3)
        waves.scale(2)
        waves.arrange(DOWN, buff=1.0)
        labels = VGroup()
        for n, wave in zip(it.count(1), waves):
            wave.stretch(0.5 * n, 1)
            label = TexText(f"{n} $hf$" + ("" if n > 1 else ""))
            label.scale(0.5)
            label.next_to(wave, UP, buff=0.2)
            labels.add(label)
            self.play(
                FadeIn(wave),
                FadeIn(label),
            )

        dots = Tex(R"\vdots", font_size=60)
        dots.next_to(waves, DOWN)
        self.play(Write(dots))
        self.wait()


class ContinuousGraph(InteractiveScene):
    def construct(self):
        axes = Axes((0, 5), (0, 5), width=4, height=4)
        graph = axes.get_graph(lambda x: (x**2) / 5)
        graph.set_stroke(YELLOW, 3)
        self.add(axes)
        self.play(ShowCreation(graph))
        self.wait()


class DiscreteGraph(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes((0, 5), (0, 5), width=4, height=4)
        graph = axes.get_graph(
            lambda x: np.floor(x) + 0.5,
            discontinuities=np.arange(0, 8),
            x_range=(0, 4.99),
        )
        graph.set_stroke(RED, 5)
        self.add(axes)
        self.play(ShowCreation(graph))
        self.wait()



class RedFilter(InteractiveScene):
    def construct(self):
        image = ImageMobject()
        image.set_height(FRAME_HEIGHT)

        # self.add(image)

        plane = NumberPlane()
        plane.fade(0.75)

        # Pixel indicator
        indicator = self.get_pixel_indicator(image)
        indicator.move_to(plane.c2p(5.25, -0.25), DOWN)

        self.add(indicator)

        # Move around
        self.wait()
        self.play(
            indicator.animate.move_to(plane.c2p(1.5, -0.125), DOWN),
            run_time=6,
        )
        self.wait()
        self.play(
            indicator.animate.move_to(plane.c2p(-3.5, 0), DOWN),
            run_time=6,
        )
        self.wait()

    def get_pixel_indicator(self, image, vect_len=2.0, direction=DOWN, square_size=1.0):
        vect = Vector(vect_len * direction, stroke_color=WHITE)
        square = Square(side_length=square_size)
        square.set_stroke(WHITE, 1)
        square.next_to(vect, -direction)

        def get_color():
            points = vect.get_end() + 0.05 * compass_directions(12)
            rgbs = np.array([image.point_to_rgb(point) for point in points])
            return rgb_to_color(rgbs.mean(0))

        square.add_updater(lambda s: s.set_fill(get_color(), 1))
        return VGroup(square, vect)


class LengthsOnDifferentColors(InteractiveScene):
    def construct(self):
        # Setup
        image = ImageMobject()
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        plane = NumberPlane()
        plane.fade(0.75)
        # self.add(plane)

        # Rectangles
        rects = Rectangle().replicate(4)
        rects.set_stroke(width=0)
        rects.set_fill(BLACK, opacity=0)
        rects.set_height(2)
        rects.set_width(FRAME_WIDTH, stretch=True)
        rects.arrange(DOWN, buff=0)

        # Braces
        lines = VGroup(
            Line(plane.c2p(3.2, 2.8), plane.c2p(-0.5, 2.8)),
            Line(plane.c2p(3.2, 0.9), plane.c2p(0.6, 0.9)),
            Line(plane.c2p(3.2, -1.0), plane.c2p(1.0, -1.0)),
            Line(plane.c2p(3.2, -3.0), plane.c2p(1.3, -3.0)),
        )
        braces = VGroup(*(
            Brace(line, DOWN, buff=SMALL_BUFF)
            for line in lines
        ))
        braces.set_backstroke(BLACK, 3)
        numbers = VGroup(*(
            DecimalNumber(line.get_length() / 7, font_size=36, unit=R" \text{m}")
            for line in lines
        ))
        for number, brace in zip(numbers, braces):
            number.next_to(brace, DOWN, buff=SMALL_BUFF)

        # Show braces
        for brace, rect, number in zip(braces, rects, numbers):
            globals().update(locals())
            other_rects = VGroup(*(r for r in rects if r is not rect))
            self.play(
                GrowFromPoint(brace, brace.get_right()),
                CountInFrom(number, 0),
                UpdateFromFunc(VGroup(), lambda m: number.next_to(brace, DOWN, SMALL_BUFF)),
                rect.animate.set_opacity(0),
                other_rects.animate.set_opacity(0.8),
            )
            self.wait(2)
        self.play(FadeOut(rects))

        # Ribbons
        axes_3d = ThreeDAxes((0, 6))
        ribbons = Group()
        twist_rates = [1.0 / PI / line.get_length() for line in lines]
        twist_rates = [0.09, 0.11, 0.115, 0.12]
        for line, twist_rate in zip(lines, twist_rates):
            ribbon = TwistedRibbon(
                axes_3d,
                amplitude=0.25,
                twist_rate=twist_rate,
                color=rgb_to_color(image.point_to_rgb(line.get_start())),
            )
            ribbon.rotate(PI / 2, RIGHT)
            ribbon.set_opacity(0.75)
            ribbon.flip(UP)
            ribbon.next_to(line, UP, MED_LARGE_BUFF, aligned_edge=RIGHT)
            ribbons.add(ribbon)

        for ribbon in ribbons:
            self.play(ShowCreation(ribbon, run_time=2))
        self.wait()


class AskAboutDiagonal(InteractiveScene):
    def construct(self):
        randy = Randolph(height=2)
        self.play(randy.says("Why diagonal?", mode="maybe", look_at=DOWN))
        self.play(Blink(randy))
        self.wait()


class AskNoteVerticalVariation(RedFilter):
    def construct(self):
        image = ImageMobject()
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        plane = NumberPlane()
        plane.fade(0.75)
        # self.add(plane)

        indicator = self.get_pixel_indicator(image)

        # Scan horizontally, then vertically
        indicator.move_to(plane.c2p(-0.5, -0.5), DOWN)

        lr_arrows = VGroup(Vector(LEFT), Vector(RIGHT))
        lr_arrows.arrange(RIGHT, buff=1.0)
        lr_arrows.move_to(plane.c2p(0.5, -1.5))
        ud_arrows = VGroup(Vector(UP), Vector(DOWN))
        ud_arrows.arrange(DOWN, buff=1.0)
        ud_arrows.move_to(plane.c2p(0, -0.6))

        self.add(indicator)
        self.play(
            FadeIn(lr_arrows, time_span=(0, 1)),
            indicator.animate.move_to(plane.c2p(3, -0.5), DOWN),
            run_time=3
        )
        self.play(indicator.animate.move_to(plane.c2p(1.5, -0.5), DOWN), run_time=3)
        self.wait()
        self.play(
            FadeIn(ud_arrows, time_span=(0, 1)),
            FadeOut(lr_arrows, time_span=(0, 1)),
            indicator.animate.move_to(plane.c2p(1.5, -0.2), DOWN),
            run_time=3
        )
        self.play(indicator.animate.move_to(plane.c2p(1.5, -0.9), DOWN), run_time=3)
        self.play(indicator.animate.move_to(plane.c2p(1.5, -0.5), DOWN), run_time=3)
        self.wait()


class CombineColors(InteractiveScene):
    def construct(self):
        # Get images
        folder = ""
        images = Group(*(
            ImageMobject()
            for ext in ["Red", "Orange", "Green", "Blue", "Rainbow"]
        ))
        colors = images[:4]
        rainbow = images[4]
        colors.set_height(FRAME_HEIGHT / 2)
        colors.arrange_in_grid(buff=0)
        rainbow.set_height(FRAME_HEIGHT)

        self.add(*colors)


class SteveMouldMention(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2)
        morty.to_edge(DOWN)
        self.play(
            morty.says("""
                This is the part that
                Steve Mould explains
                quite well, by the way
            """, mode="tease")
        )
        self.play(Blink(morty))
        self.wait()


class IndexOfRefraction(InteractiveScene):
    def construct(self):
        # Test
        equation = Tex(R"{\text{Speed in a vacuum} \over \text{Speed in water}} \approx 1.33")
        equation.shift(UP)
        rhs = equation[R"\approx 1.33"]
        equation.scale(0.75)
        rhs.scale(1 / 0.75, about_edge=LEFT)
        arrow = Vector(DOWN)
        arrow.next_to(rhs, DOWN, SMALL_BUFF)
        words = TexText("``Index of refraction''")
        words.next_to(arrow, DOWN, SMALL_BUFF)
        words.set_color(BLUE)

        self.play(FadeIn(equation, 0.5 * UP))
        self.wait()
        self.play(
            GrowArrow(arrow),
            Write(words)
        )
        self.wait()


class LayerKickBackLabel(InteractiveScene):
    def construct(self):
        layer_label = Text("Each layer kicks back the phase")
        layer_label.to_edge(UP)
        self.play(Write(layer_label))
        self.wait()


class SugarIsChiral(InteractiveScene):
    default_frame_orientation = (-33, 85)
    title = R"Sucrose $\text{C}_{12}\text{H}_{22}\text{O}_{11}$ "
    subtitle = "(D-Glucose + D-Fructose)"
    molecule_height = 3

    def construct(self):
        axes = ThreeDAxes()
        axes.set_stroke(opacity=0.5)
        # self.add(axes)

        # Set up
        frame = self.frame
        title = VGroup(
            TexText(self.title),
            TexText(self.subtitle),
        )
        title[1].scale(0.7).set_color(GREY_A)
        title.arrange(DOWN)
        title.fix_in_frame()
        title.to_edge(UP, buff=0.25)

        sucrose = Sucrose()
        sucrose.set_height(self.molecule_height)

        # Introduce
        frame.reorient(24, 74, 0)
        self.add(title)
        self.play(
            FadeIn(sucrose, scale=5),
        )
        self.play(
            self.frame.animate.reorient(-16, 75, 0).set_anim_args(run_time=6)
        )
        self.add(sucrose)
        self.wait()

        # Show mirror image
        mirror = Square(side_length=6)
        mirror.set_fill(BLUE, 0.35)
        mirror.set_stroke(width=0)
        mirror.rotate(PI / 2, UP)
        mirror.set_shading(0, 1, 0)
        mirror.stretch(3, 1)

        sucrose.target = sucrose.generate_target()
        sucrose.target.next_to(mirror, LEFT, buff=1.0)

        self.add(mirror, sucrose)
        self.play(
            frame.animate.move_to(0.75 * OUT),
            MoveToTarget(sucrose),
            FadeIn(mirror),
            title.animate.scale(0.75).match_x(sucrose.target).set_y(1.75),
            run_time=1.5,
        )

        mirror_image = sucrose.copy()
        mirror_image.target = mirror_image.generate_target()
        mirror_image.target.stretch(-1, 0, about_point=mirror.get_center())

        mirror_words = Text("(mirror image)", font_size=36, color=GREY_A)
        mirror_words.fix_in_frame()
        mirror_words.match_x(mirror_image.target)
        mirror_words.match_y(title)

        self.add(mirror_image, mirror, sucrose)
        self.play(
            MoveToTarget(mirror_image),
            FadeIn(mirror_words),
        )

        # Chiral definition
        definition = TexText(R"Chiral $\rightarrow$ Cannot be superimposed onto its mirror image")
        definition.fix_in_frame()
        definition.to_edge(UP)

        sucrose.add_updater(lambda m, dt: m.rotate(10 * DEGREES * dt, axis=OUT))
        mirror_image.add_updater(lambda m, dt: m.rotate(-10 * DEGREES * dt, axis=OUT))
        self.play(Write(definition))
        self.play(
            self.frame.animate.reorient(-8, 76, 0),
            run_time=15,
        )
        self.wait(15)


class SimplerChiralShape(InteractiveScene):
    default_frame_orientation = (0, 70)

    def construct(self):
        # Ribbon
        frame = self.frame
        frame.set_field_of_view(1 * DEGREES)
        axes = ThreeDAxes((-3, 3))
        ribbon = TwistedRibbon(axes, amplitude=1, twist_rate=-0.35)
        ribbon.rotate(PI / 2, DOWN)
        ribbon.set_color(RED)
        ribbon.set_opacity(0.9)
        ribbon.set_shading(0.5, 0.5, 0.5)
        always(ribbon.sort_faces_back_to_front, UP)
        ribbon.set_x(-4)

        mirror_image = ribbon.copy()
        mirror_image.stretch(-1, 0)
        mirror_image.set_x(-ribbon.get_x())
        mirror_image.set_color(YELLOW_C)
        mirror_image.set_opacity(0.9)

        # Title
        spiral_name = Text("Spiral")
        mirror_name = Text("Mirror image")
        for name, mob in [(spiral_name, ribbon), (mirror_name, mirror_image)]:
            name.fix_in_frame()
            name.to_edge(UP)
            name.match_x(mob)

        self.play(
            FadeIn(spiral_name, 0.5 * UP),
            ShowCreation(ribbon, run_time=3),
        )
        self.wait()
        self.play(
            ReplacementTransform(ribbon.copy().shift(0.1 * DOWN), mirror_image),
            FadeTransformPieces(spiral_name.copy(), mirror_name),
        )
        self.wait()

        # Reorient
        r_copy = ribbon.copy()
        self.play(r_copy.animate.next_to(mirror_image, LEFT))
        self.play(Rotate(r_copy, PI, RIGHT, run_time=2))
        self.play(Rotate(r_copy, PI, OUT, run_time=2))
        self.play(Rotate(r_copy, PI, UP, run_time=2))
        self.wait()


class SucroseAction(InteractiveScene):
    just_sucrose = False

    def construct(self):
        # Sucrose
        sucrose = Sucrose(height=1.5)
        sucrose.balls.scale_radii(0.5)
        sucrose.rotate(PI / 2, RIGHT)
        sucrose.to_edge(LEFT)
        sucrose.add_updater(lambda m, dt: m.rotate(10 * DEGREES * dt, UP))
        if self.just_sucrose:
            self.add(sucrose)
            self.wait(36)
            return

        # Arrows
        arrows = VGroup()
        words = VGroup(
            Text("The amount that sugar\nslows this light..."),
            Text("...is different from how\nit slows this light"),
        )
        words.scale(0.75)
        for sign, word in zip([+1, -1], words):
            arrow = Line(ORIGIN, 5 * RIGHT + 1.5 * sign * UP, path_arc=-sign * 60 * DEGREES)
            arrow.insert_n_curves(100)
            arrow.add_tip(length=0.2, width=0.2)
            arrow.shift(sucrose.get_right() + 0.5 * LEFT + sign * 1.25 * UP)
            arrows.add(arrow)
            word.next_to(arrow, sign * UP, buff=0.1)

            self.play(
                ShowCreation(arrow),
                FadeIn(word, 0.25 * sign * UP)
            )
            self.wait(2)


class SucroseActionSucrosePart(SucroseAction):
    just_sucrose = True


class ThatSeemsIrrelevant(InteractiveScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(
            stds[0].says("That seems irrelevant", mode="sassy"),
            self.change_students(None, "erm", "hesitant", look_at=stds[0].eyes),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            stds[0].debubble(mode="raise_left_hand", look_at=self.screen),
            self.change_students(None, "pondering", "pondering", look_at=self.screen),
        )
        self.wait(5)


class BigPlus(InteractiveScene):
    def construct(self):
        brace = Brace(Line(2 * DOWN, 2 * UP), RIGHT)
        brace.set_height(7)
        brace.move_to(2 * LEFT)
        plus = Tex("+", font_size=90)
        plus.next_to(brace, LEFT, buff=2.5)
        equals = Tex("=", font_size=90)
        equals.next_to(brace, RIGHT, buff=1.0)

        self.add(plus, brace, equals)


class CurvyCurvyArrow(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(path_arc=-PI, buff=0.2, stroke_width=30, tip_width_ratio=4)
        arrows = VGroup(
            Arrow(DOWN, UP, **kw),
            Arrow(UP, DOWN, **kw),
        )
        arrows.set_color(WHITE)
        arrows.set_height(5)
        self.frame.reorient(-9, 75, 90)
        self.frame.set_field_of_view(1 * DEGREES)
        self.add(arrows)


class GlowDotScene(InteractiveScene):
    def construct(self):
        self.add(GlowDot(radius=3, color=WHITE))

        mid_point = 0.85 * LEFT
        mask = VMobject().set_points_as_corners([
            UR, mid_point, DR, DL, UL,
        ])
        mask.set_fill(BLACK, 1)
        mask.set_stroke(width=0)
        mask.set_height(20, about_point=mid_point)
        self.add(mask)


class Randy(InteractiveScene):
    def construct(self):
        self.add(Randolph(mode="confused"))


class SimpleLightBeam(InteractiveScene):
    default_frame_orientation = (-33, 85)
    axes_config = dict()
    z_amplitude = 0.5
    wave_len = 2.0
    speed = 1.0
    color = YELLOW
    oscillating_field_config = dict(
        stroke_opacity=0.5,
        stroke_width=2,
        tip_width_ratio=1
    )

    def construct(self):
        axes, plane = get_axes_and_plane(**self.axes_config)
        self.add(axes, plane)

        # Introduce wave
        wave = OscillatingWave(
            axes,
            z_amplitude=self.z_amplitude,
            wave_len=self.wave_len,
            speed=self.speed,
            color=self.color
        )
        vect_wave = OscillatingFieldWave(axes, wave, **self.oscillating_field_config)

        def update_wave(wave):
            st = self.time * self.speed  # Suppressor threshold
            points = wave.get_points().copy()
            xs = axes.x_axis.p2n(points)
            suppressors = np.clip(smooth(st - xs), 0, 1)
            points[:, 1] *= suppressors
            points[:, 2] *= suppressors
            wave.set_points(points)
            return wave

        wave.add_updater(update_wave)
        vect_wave.add_updater(update_wave)

        self.add(wave)
        self.play(
            self.frame.animate.reorient(-98, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=8,
        )
        self.add(vect_wave, wave)
        self.play(
            VFadeIn(vect_wave),
            self.frame.animate.reorient(-10, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=4
        )
        self.wait(3)

        # Label directions
        z_label = Tex("z")
        z_label.rotate(PI / 2, RIGHT)
        z_label.next_to(axes.z_axis, OUT)

        y_label = Tex("y")
        y_label.rotate(PI / 2, RIGHT)
        y_label.next_to(axes.y_axis, UP + OUT)

        x_label = VGroup(
            TexText("$x$-direction"),
            Vector(RIGHT, stroke_color=WHITE),
        )
        x_label.arrange(RIGHT)
        x_label.set_flat_stroke(False)
        x_label.rotate(PI / 2, RIGHT)
        x_label.next_to(z_label, RIGHT, buff=2.0)
        x_label.match_z(axes.c2p(0, 0, 0.75))

        self.play(
            FadeIn(z_label, 0.5 * OUT),
            FadeIn(y_label, 0.5 * UP),
        )
        self.wait(3)
        self.play(
            Write(x_label[0]),
            GrowArrow(x_label[1]),
        )
        self.play(
            self.frame.animate.reorient(-41, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=12,
        )
        self.wait(6)


class TwistingLightBeam(SimpleLightBeam):
    z_amplitude = 0.5
    wave_len = 2.0
    twist_rate = 1 / 72
    speed = 1.0
    color = YELLOW

    def construct(self):
        # Axes
        axes, plane = get_axes_and_plane(**self.axes_config)
        self.add(axes, plane)

        # Add wave
        wave = OscillatingWave(
            axes,
            z_amplitude=self.z_amplitude,
            wave_len=self.wave_len,
            speed=self.speed,
            color=self.color
        )
        vect_wave = OscillatingFieldWave(axes, wave, **self.oscillating_field_config)

        twist_rate_tracker = ValueTracker(0)

        def update_twist_rate(wave):
            wave.twist_rate = twist_rate_tracker.get_value()
            return wave

        wave.add_updater(update_twist_rate)

        cylinder = SugarCylinder(axes, self.camera, radius=self.z_amplitude)

        self.add(vect_wave, wave)
        self.frame.reorient(-41, 77, 0).move_to([-0.87, 0.9, -0.43])
        self.wait(4)
        cylinder.save_state()
        cylinder.stretch(0, 0, about_edge=RIGHT)
        self.play(
            Restore(cylinder, time_span=(0, 3)),
            twist_rate_tracker.animate.set_value(self.twist_rate).set_anim_args(time_span=(0, 3)),
            self.frame.animate.reorient(-47, 80, 0).move_to([0.06, -0.05, 0.05]).set_height(8.84),
            run_time=6,
        )
        self.wait(2)
        self.play(
            self.frame.animate.reorient(-130, 77, 0).move_to([0.35, -0.36, 0.05]),
            run_time=10,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(-57, 77, 0).move_to([0.35, -0.36, 0.05]),
            run_time=10,
        )

        # Add rod with oscillating ball
        x_tracker, plane, rod, ball, x_label = self.get_slice_group(axes, wave)
        plane.save_state()
        plane.stretch(0, 2, about_edge=OUT)

        frame_anim = self.frame.animate.reorient(-45, 79, 0)
        frame_anim.move_to([0.63, 0.47, -0.25])
        frame_anim.set_height(10.51)
        frame_anim.set_anim_args(run_time=3)

        self.add(rod, ball, plane, cylinder)
        self.play(
            frame_anim,
            FadeIn(rod),
            Restore(plane),
            FadeIn(x_label),
            UpdateFromAlphaFunc(wave,
                lambda m, a: m.set_stroke(
                    width=interpolate(2, 1, a),
                    opacity=interpolate(1, 0.5, a),
                ),
                run_time=3,
                time_span=(0, 2),
            ),
            UpdateFromAlphaFunc(ball, lambda m, a: m.set_opacity(a)),
        )
        self.wait(9)

        # Show twist down the line of the cylinder
        x_tracker.set_value(0)
        x_tracker.clear_updaters()
        x_tracker.add_updater(lambda m, dt: m.increment_value(0.5 * dt))
        self.add(x_tracker)
        self.wait(5)
        self.play(
            self.frame.animate.reorient(-87, 88, 0).move_to([0.63, 0.47, -0.25]).set_height(10.51),
            run_time=5,
        )
        self.wait(3)
        self.play(
            self.frame.animate.reorient(-43, 78, 0).move_to([0.63, 0.47, -0.25]).set_height(10.51),
            run_time=5
        )
        self.play(
            self.frame.animate.reorient(-34, 80, 0).move_to([1.61, -0.05, 0.3]).set_height(10.30),
            run_time=15,
        )
        self.wait(10)

    def get_slice_group(self, axes, wave):
        x_tracker = ValueTracker(0)
        get_x = x_tracker.get_value

        rod = self.get_polarization_rod(axes, wave, get_x)
        ball = self.get_wave_ball(wave, get_x)
        plane = self.get_slice_plane(axes, get_x)
        x_label = self.get_plane_label(axes, plane)

        return Group(x_tracker, plane, rod, ball, x_label)

    def get_polarization_rod(self, axes, wave, get_x, stroke_color=None, length_mult=2.0, stroke_width=3):
        rod = Line(IN, OUT)
        rod.set_stroke(
            color=stroke_color or wave.get_stroke_color(),
            width=stroke_width,
        )
        rod.set_flat_stroke(False)
        wave_z = axes.z_axis.p2n(wave.get_center())
        wave_y = axes.y_axis.p2n(wave.get_center())

        def update_rod(rod):
            x = get_x()
            rod.put_start_and_end_on(
                axes.c2p(x, wave_y, wave_z - length_mult * wave.z_amplitude),
                axes.c2p(x, wave_y, wave_z + length_mult * wave.z_amplitude),
            )
            rod.rotate(TAU * wave.twist_rate * x, RIGHT)
            return rod

        rod.add_updater(update_rod)
        return rod

    def get_wave_ball(self, wave, get_x, radius=0.075):
        ball = TrueDot(radius=radius)
        ball.make_3d()
        ball.set_color(wave.get_color())

        def update_ball(ball):
            ball.move_to(wave.offset + wave.xt_to_point(get_x(), wave.time))
            return ball

        ball.add_updater(update_ball)
        return ball

    def get_slice_plane(self, axes, get_x):
        plane = Square(side_length=axes.z_axis.get_length())
        plane.set_fill(BLUE, 0.25)
        plane.set_stroke(width=0)
        circle = Circle(
            radius=axes.z_axis.get_unit_size() * self.z_amplitude,
            n_components=100,
        )
        circle.set_flat_stroke(False)
        circle.set_stroke(BLACK, 1)
        plane.add(circle)
        plane.rotate(PI / 2, UP)
        plane.add_updater(lambda m: m.move_to(axes.c2p(get_x(), 0, 0)))
        return plane

    def get_plane_label(self, axes, plane, font_size=24, color=GREY_B):
        x_label = Tex("x = 0.00", font_size=font_size)
        x_label.set_fill(color)
        x_label.value_mob = x_label.make_number_changable("0.00")
        x_label.rotate(PI / 2, RIGHT)
        x_label.rotate(PI / 2, IN)

        def update_x_label(x_label):
            x_value = x_label.value_mob
            x_value.set_value(axes.x_axis.p2n(plane.get_center()))
            x_value.rotate(PI / 2, RIGHT)
            x_value.rotate(PI / 2, IN)
            x_value.next_to(x_label[1], DOWN, SMALL_BUFF)
            x_label.next_to(plane, OUT)
            return x_label

        x_label.add_updater(update_x_label)
        return x_label


class TwistingBlueLightBeam(TwistingLightBeam):
    wave_len = 1.0
    twist_rate = 1 / 48
    color = PURPLE


class TwistingRedLightBeam(TwistingLightBeam):
    wave_len = 3.0
    twist_rate = 1 / 96
    color = RED


class TwistingWithinCylinder(InteractiveScene):
    default_frame_orientation = (-40, 80)
    n_lines = 11
    pause_down_the_tube = True

    def construct(self):
        # Reference objects
        frame = self.frame
        axes, plane = get_axes_and_plane(
            x_range=(0, 8),
            y_range=(-2, 2),
            z_range=(-2, 2),
            y_unit=1,
            z_unit=1,
            origin_point=3 * LEFT
        )
        cylinder = SugarCylinder(axes, self.camera, radius=0.5)

        self.add(plane, axes)
        self.add(cylinder)

        # Light lines
        lines = VGroup()
        colors = get_spectral_colors(self.n_lines)
        for color in colors:
            line = Line(ORIGIN, 0.95 * OUT)
            line.set_flat_stroke(False)
            line.set_stroke(color, 2)
            lines.add(line)

        lines.arrange(DOWN, buff=0.1)
        lines.move_to(cylinder.get_left())

        # Add polarizer to the start
        light = GlowDot(color=WHITE, radius=3)
        light.move_to(axes.c2p(-3, 0, 0))
        polarizer = Polarizer(axes, radius=0.6)
        polarizer.move_to(axes.c2p(-1, 0, 0))
        polarizer_label = Text("Linear polarizer", font_size=36)
        polarizer_label.rotate(PI / 2, RIGHT)
        polarizer_label.rotate(PI / 2, IN)
        polarizer_label.next_to(polarizer, OUT)
        frame.reorient(-153, 79, 0)
        frame.shift(1.0 * IN)

        self.play(GrowFromCenter(light))
        self.play(
            Write(polarizer_label),
            FadeIn(polarizer, IN),
            light.animate.shift(LEFT).set_anim_args(time_span=(1, 3)),
            self.frame.animate.reorient(-104, 77, 0).center().set_anim_args(run_time=3),
        )

        # Many waves
        waves = VGroup(*(
            OscillatingWave(
                axes,
                z_amplitude=0.3,
                wave_len=wave_len,
                color=line.get_color(),
                offset=LEFT + line.get_y() * UP
            )
            for line, wave_len in zip(
                lines,
                np.linspace(2.0, 0.5, len(lines))
            )
        ))
        waves.set_stroke(width=1)
        superposition = MeanWave(waves)
        superposition.set_stroke(WHITE, 2)
        superposition.add_updater(lambda m: m.stretch(4, 2, about_point=ORIGIN))

        self.play(
            VFadeIn(superposition),
            FadeOut(cylinder),
        )
        self.play(
            self.frame.animate.reorient(-66, 76, 0),
            light.animate.scale(0.25),
            run_time=10,
        )
        self.remove(superposition)
        superposition.suspend_updating()
        self.play(*(
            TransformFromCopy(superposition, wave, run_time=2)
            for wave in waves
        ))

        # Go through individual waves
        self.add(waves)
        for wave1 in waves:
            anims = []
            for wave2 in waves:
                wave2.current_opacity = wave2.get_stroke_opacity()
                if wave1 is wave2:
                    wave2.target_opacity = 1
                else:
                    wave2.target_opacity = 0.1
                anims.append(UpdateFromAlphaFunc(wave2, lambda m, a: m.set_stroke(
                    opacity=interpolate(m.current_opacity, m.target_opacity, a)
                )))
            self.play(*anims, run_time=0.5)
            self.wait()

        for wave in waves:
            wave.current_opacity = wave.get_stroke_opacity()
            wave.target_opacity = 1

        self.play(
            *(
                UpdateFromAlphaFunc(wave, lambda m, a: m.set_stroke(
                    opacity=interpolate(m.current_opacity, m.target_opacity, a)
                ))
                for wave in waves
            ),
            frame.animate.reorient(-55, 76, 0).move_to([-0.09, 0.13, -0.17]).set_height(7.5),
            run_time=3
        )

        # Introduce lines
        white_lines = lines.copy()
        white_lines.set_stroke(WHITE)
        white_lines.arrange(UP, buff=0)
        white_lines.move_to(axes.get_origin())

        plane = Square(side_length=2 * axes.z_axis.get_unit_size())
        plane.set_fill(WHITE, 0.25)
        plane.set_stroke(width=0)
        plane.add(
            Circle(radius=0.5 * cylinder.get_depth(), n_components=100).set_stroke(BLACK, 1)
        )
        plane.rotate(PI / 2, UP)
        plane.move_to(axes.get_origin())
        plane.save_state()
        plane.stretch(0, 2, about_edge=UP)

        self.play(
            ReplacementTransform(waves, lines, lag_ratio=0.1, run_time=3),
            frame.animate.reorient(-61, 83, 0).move_to([0.03, -0.16, -0.28]).set_height(7).set_anim_args(run_time=2),
            Restore(plane),
            FadeIn(cylinder),
        )
        self.add(axes, lines)
        self.wait()
        self.play(
            lines.animate.arrange(UP, buff=0).move_to(axes.get_origin()),
            FadeIn(white_lines),
            FadeOut(polarizer),
            FadeOut(polarizer_label),
            FadeOut(light),
        )
        self.wait()

        # Enable lines to twist through the tube
        line_start, line_end = white_lines[0].get_start_and_end()

        distance_tracker = ValueTracker(0)

        wave_lengths = np.linspace(700, 400, self.n_lines)  # Is this right?
        for line, wave_length in zip(lines, wave_lengths):
            line.wave_length = wave_length

        def update_lines(lines):
            dist = distance_tracker.get_value()
            for line in lines:
                line.set_points_as_corners([line_start, line_end])
                line.rotate(get_twist(line.wave_length, dist), RIGHT)
                line.move_to(axes.c2p(dist, 0, 0))
                line.set_gloss(3 * np.exp(-3 * dist))

        lines.add_updater(update_lines)

        # Add wave trails
        trails = VGroup(*(
            self.get_wave_trail(line)
            for line in lines
        ))
        continuous_trails = Group(*(
            self.get_continuous_wave_trail(axes, line)
            for line in lines
        ))
        for trail in continuous_trails:
            x_unit = axes.x_axis.get_unit_size()
            x0 = axes.get_origin()[0]
            trail.add_updater(
                lambda t: t.set_clip_plane(LEFT, distance_tracker.get_value() + x0)
            )
        self.add(trails, lines, white_lines)

        # Move light beams down the pole
        self.add(distance_tracker)
        distance_tracker.set_value(0)
        plane.add_updater(lambda m: m.match_x(lines))
        self.remove(white_lines)

        if self.pause_down_the_tube:
            # Test
            self.play(
                self.frame.animate.reorient(-42, 76, 0).move_to([0.03, -0.16, -0.28]).set_height(7.00),
                distance_tracker.animate.set_value(4),
                run_time=6,
                rate_func=linear,
            )
            trails.suspend_updating()
            self.play(
                self.frame.animate.reorient(67, 77, 0).move_to([-0.31, 0.48, -0.33]).set_height(4.05),
                run_time=3,
            )
            self.wait(2)
            trails.resume_updating()
            self.play(
                distance_tracker.animate.set_value(axes.x_axis.x_max),
                self.frame.animate.reorient(-36, 79, 0).move_to([-0.07, 0.06, 0.06]).set_height(7.42),
                run_time=6,
                rate_func=linear,
            )
            trails.clear_updaters()
            self.play(
                self.frame.animate.reorient(-10, 77, 0).move_to([0.42, -0.16, -0.03]).set_height(5.20),
                trails.animate.set_stroke(width=3, opacity=0.25).set_anim_args(time_span=(0, 3)),
                run_time=10,
            )
        else:
            self.play(
                self.frame.animate.reorient(-63, 84, 0).move_to([1.04, -1.86, 0.55]).set_height(1.39),
                distance_tracker.animate.set_value(axes.x_axis.x_max),
                run_time=15,
                rate_func=linear,
            )
            trails.clear_updaters()
            lines.clear_updaters()

            self.play(
                self.frame.animate.reorient(64, 81, 0).move_to([3.15, 0.46, -0.03]).set_height(5),
                run_time=3,
            )
            self.wait()

        # Add polarizer at the end
        end_polarizer = Polarizer(axes, radius=0.6)
        end_polarizer.next_to(lines, RIGHT, buff=0.5)

        self.play(
            FadeIn(end_polarizer, OUT),
            FadeOut(plane),
            self.frame.animate.reorient(54, 78, 0).move_to([3.15, 0.46, -0.03]).set_height(5.00).set_anim_args(run_time=4)
        )
        end_polarizer.save_state()
        self.play(end_polarizer.animate.fade(0.8))

        # Show a few different frequencies
        vertical_components = VGroup()
        for index in range(len(lines)):
            lines.generate_target()
            trails.generate_target()
            lines.target.set_opacity(0)
            trails.target.set_opacity(0)
            lines.target[index].set_opacity(1)
            trails.target[index].set_opacity(0.2)

            line = lines[index]
            x = float(axes.x_axis.p2n(cylinder.get_right()))
            vcomp = line.copy().set_opacity(1)
            vcomp.stretch(0, 1)
            vcomp.move_to(axes.c2p(x, -2 + index / len(lines), 0))
            z = float(axes.z_axis.p2n(vcomp.get_zenith()))
            y_min, y_max = axes.y_range[:2]
            globals().update(locals())
            dashed_lines = VGroup(*(
                DashedLine(axes.c2p(x, y_min, u * z), axes.c2p(x, y_max, u * z), dash_length=0.02)
                for u in [1, -1]
            ))
            dashed_lines.set_stroke(WHITE, 0.5)
            dashed_lines.set_flat_stroke(False)

            self.play(
                MoveToTarget(lines),
                MoveToTarget(trails),
                FadeIn(dashed_lines),
                FadeIn(vcomp),
                self.frame.animate.reorient(77, 87, 0).move_to([3.1, 0.4, 0]).set_height(5),
            )
            self.play(
                FadeOut(dashed_lines),
            )

            vertical_components.add(vcomp)

        self.play(
            lines.animate.set_opacity(1),
            trails.animate.set_opacity(0.05),
        )

        # Final color
        def get_final_color():
            rgbs = np.array([
                line.data["stroke_rgba"][0, :3]
                for line in lines
            ])
            depths = np.array([v_line.get_depth() for v_line in vertical_components])
            alphas = depths / depths.sum()
            rgb = ((rgbs**0.5) * alphas[:, np.newaxis]).sum(0)**2.0
            return rgb_to_color(rgb)

        new_color = get_final_color()
        new_lines = vertical_components.copy()
        for line in new_lines:
            line.set_depth(cylinder.get_depth())
            line.set_stroke(new_color, 4)
            line.next_to(end_polarizer, RIGHT, buff=0.5)

        self.play(
            Restore(end_polarizer),
            TransformFromCopy(vertical_components, new_lines),
            self.frame.animate.reorient(43, 73, 0).move_to([3.3, 0.66, -0.38]).set_height(5.68),
            run_time=4,
        )
        self.play(
            self.frame.animate.reorient(45, 72, 0).move_to([3.17, 0.4, -0.56]),
            run_time=8,
        )

        # Twist the tube
        result_line = new_lines[0]
        self.remove(new_lines)
        self.add(result_line)
        result_line.add_updater(lambda l: l.set_stroke(get_final_color()))

        line_group = VGroup(trails, lines)

        p1, p2 = axes.c2p(0, 1, 0), axes.c2p(0, -1, 0)
        twist_arrows = VGroup(
            Arrow(p1, p2, path_arc=PI),
            Arrow(p2, p1, path_arc=PI),
        )
        twist_arrows.rotate(PI / 2, UP, about_point=axes.get_origin())
        twist_arrows.apply_depth_test()
        self.add(twist_arrows, cylinder, line_group, vertical_components)

        for v_comp, line in zip(vertical_components, lines):
            v_comp.line = line
            v_comp.add_updater(lambda m: m.match_depth(m.line))

        self.play(
            ShowCreation(twist_arrows, lag_ratio=0),
            Rotate(line_group, PI, axis=RIGHT, run_time=12, rate_func=linear)
        )

    def get_wave_trail(self, line, spacing=0.05, opacity=0.05):
        trail = VGroup()
        trail.time = 1

        def update_trail(trail, dt):
            trail.time += dt
            if trail.time > spacing:
                trail.time = 0
                trail.add(line.copy().set_opacity(opacity).set_shading(0, 0, 0))

        trail.add_updater(update_trail)
        return trail

    def get_continuous_wave_trail(self, axes, line, opacity=0.4):
        return TwistedRibbon(
            axes,
            amplitude=0.5 * line.get_length(),
            twist_rate=get_twist(line.wave_length, TAU),
            color=line.get_color(),
            opacity=opacity,
        )


class InducedWiggleInCylinder(TwistingLightBeam):
    random_seed = 3
    cylinder_radius = 0.5
    wave_config = dict(
        z_amplitude=0.15,
        wave_len=0.5,
        color=get_spectral_color(0.1),
        speed=1.0,
        twist_rate=-1 / 24
    )

    def construct(self):
        # Setup
        frame = self.frame
        frame.reorient(-51, 80, 0).move_to(0.5 * IN).set_height(9)

        axes, plane = get_axes_and_plane(**self.axes_config)
        cylinder = SugarCylinder(axes, self.camera, radius=self.cylinder_radius)
        wave = OscillatingWave(axes, **self.wave_config)
        x_tracker, plane, rod, ball, x_label = slice_group = self.get_slice_group(axes, wave)
        rod = self.get_polarization_rod(axes, wave, x_tracker.get_value, length_mult=5.0)

        axes_labels = Tex("yz", font_size=30)
        axes_labels.rotate(89 * DEGREES, RIGHT)
        axes_labels[0].next_to(axes.y_axis.get_top(), OUT, SMALL_BUFF)
        axes_labels[1].next_to(axes.z_axis.get_zenith(), OUT, SMALL_BUFF)
        axes.add(axes_labels)

        light = GlowDot(radius=4, color=RED)
        light.move_to(axes.c2p(-3, 0, 0))

        polarizer = Polarizer(axes, radius=0.5)
        polarizer.move_to(axes.c2p(-1, 0, 0))

        self.add(axes, cylinder, polarizer, light)

        # Bounces of various points
        randy = self.get_observer(axes.c2p(8, -3, -0.5))
        self.play(
            self.frame.animate.reorient(-86, 70, 0).move_to([1.01, -2.98, -0.79]).set_height(11.33),
            FadeIn(randy, time_span=(0, 1)),
            run_time=2,
        )
        max_y = 0.5 * self.cylinder_radius
        line = VMobject()
        line.set_stroke(RED, 2)
        line.set_flat_stroke(False)
        dot = TrueDot(radius=0.05)
        dot.make_3d()
        for x in range(10):
            point = axes.c2p(
                random.uniform(axes.x_axis.x_min, axes.x_axis.x_max),
                random.uniform(-max_y, -max_y),
                random.uniform(-max_y, -max_y),
            )
            line_points = [light.get_center(), point, randy.eyes.get_top()]
            self.add(dot, cylinder)
            if x == 0:
                dot.move_to(point)
                line.set_points_as_corners(line_points)
                self.play(ShowCreation(line))
            else:
                self.play(
                    line.animate.set_points_as_corners(line_points),
                    dot.animate.move_to(point),
                )
                self.wait()
        self.play(
            FadeOut(line),
            FadeOut(dot),
        )

        # Show slice such that wiggling is in z direction
        x_tracker.set_value(0)
        self.add(wave, cylinder)
        self.play(
            self.frame.animate.reorient(-73, 78, 0).move_to([0.8, -2.22, -0.83]).set_height(10.64),
            light.animate.scale(0.5),
            polarizer.animate.fade(0.5),
            VFadeIn(wave),
        )
        self.wait(4)
        self.add(wave, cylinder)
        self.play(
            FadeIn(plane),
            FadeIn(x_label),
            FadeIn(rod),
        )
        self.play(
            x_tracker.animate.set_value(12),
            run_time=12,
            rate_func=linear,
        )
        self.add(rod, ball, wave, cylinder)

        # Show observer
        line_of_sight = DashedLine(randy.eyes.get_top(), rod.get_center())
        line_of_sight.set_stroke(WHITE, 2)
        line_of_sight.set_flat_stroke(False)

        self.play(
            self.frame.animate.reorient(-60, 79, 0).move_to([0.73, -0.59, -0.39]).set_height(9.63),
            Write(line_of_sight, time_span=(3, 4), lag_ratio=0),
            run_time=5,
        )
        self.wait(2)

        # Show propagating rings
        self.show_propagation(rod)

        # Move to a less favorable spot
        new_line_of_sight = DashedLine(randy.eyes.get_top(), axes.c2p(6, 0, 0))
        new_line_of_sight.match_style(line_of_sight)
        new_line_of_sight.set_flat_stroke(False)

        self.remove(ball)
        self.play(
            x_tracker.animate.set_value(6),
            FadeOut(line_of_sight, time_span=(0, 0.5)),
            run_time=4,
        )
        self.add(ball, wave, cylinder, plane)
        self.play(ShowCreation(new_line_of_sight))
        self.wait(4)

        # New propagations
        self.show_propagation(rod)

        # Show ribbon
        ribbon = TwistedRibbon(
            axes,
            amplitude=wave.z_amplitude,
            twist_rate=wave.twist_rate,
            color=wave.get_color(),
        )

        self.add(ribbon, cylinder)
        self.play(ShowCreation(ribbon, run_time=5))
        self.wait()
        self.play(
            self.frame.animate.reorient(8, 77, 0).move_to([2.01, -0.91, -0.58]).set_height(5.55),
            FadeOut(randy),
            run_time=2,
        )
        self.wait(4)
        self.play(
            self.frame.animate.reorient(-25, 76, 0).move_to([4.22, -1.19, -0.5]),
            x_tracker.animate.set_value(12),
            FadeOut(new_line_of_sight, time_span=(0, 0.5)),
            run_time=3,
        )
        self.wait(4)
        self.play(
            self.frame.animate.reorient(-61, 78, 0).move_to([0.7, 0.05, -0.69]).set_height(9.68),
            FadeIn(randy),
            run_time=3,
        )
        self.play(
            LaggedStartMap(FadeOut, Group(
                line_of_sight, plane, rod, ball, x_label
            ))
        )

        # Show multiple waves
        n_waves = 11
        amp = 0.03
        zs = np.linspace(0.5 - amp, -0.5 + amp, n_waves)
        small_wave_config = dict(self.wave_config)
        small_wave_config["z_amplitude"] = amp

        waves = VGroup(*(
            OscillatingWave(
                axes,
                offset=axes.c2p(0, 0, z)[2] * OUT,
                **small_wave_config
            )
            for z in zs
        ))

        self.remove(ribbon)
        self.play(
            FadeOut(wave),
            VFadeIn(waves),
        )
        self.wait(4)

        # Focus on various x_slices
        x_tracker.set_value(0)
        rods = VGroup(*(
            self.get_polarization_rod(
                axes, lil_wave, x_tracker.get_value,
                length_mult=1,
                stroke_width=2,
            )
            for lil_wave in waves
        ))
        balls = Group(*(
            self.get_wave_ball(lil_wave, x_tracker.get_value, radius=0.025)
            for lil_wave in waves
        ))
        sf = 1.2 * axes.z_axis.get_unit_size() / plane.get_height()
        plane.scale(sf)
        plane[0].scale(1.0 / sf)

        plane.update()
        x_label.update()
        self.add(plane, rods, balls, cylinder, x_label)
        self.play(
            self.frame.animate.reorient(-90, 83, 0).move_to([0.17, -0.37, -0.63]).set_height(7.35).set_anim_args(run_time=3),
            FadeOut(light),
            FadeOut(polarizer),
            FadeIn(plane),
            FadeIn(rods),
            FadeIn(x_label),
            waves.animate.set_stroke(width=0.5, opacity=0.5).set_anim_args(time_span=(1, 2), suspend_mobject_updating=False),
            cylinder.animate.set_opacity(0.05).set_anim_args(time_span=(1, 2))
        )
        self.wait(4)
        self.play(
            self.frame.animate.reorient(-91, 90, 0).move_to([-0.01, -1.39, 0.21]).set_height(3.70),
            x_tracker.animate.set_value(5).set_anim_args(rate_func=linear),
            run_time=12,
        )
        self.wait(4)

        # Show lines of sight
        lines_of_sight = VGroup(*(
            self.get_line_of_sign(rod, randy, stroke_width=0.5)
            for rod in rods
        ))

        self.play(ShowCreation(lines_of_sight[0]))
        self.show_propagation(rods[0])
        for line1, line2 in zip(lines_of_sight, lines_of_sight[1:]):
            self.play(FadeOut(line1), FadeIn(line2), run_time=0.25)
            self.wait(0.25)
        self.wait(4)
        self.play(FadeIn(lines_of_sight[:-1]))
        self.add(lines_of_sight)

        # Move closer and farther
        self.play(
            randy.animate.shift(3.5 * UP + 0.5 * IN),
            run_time=2,
        )
        self.wait(8)
        self.play(
            self.frame.animate.reorient(-91, 89, 0).move_to([-0.05, -3.75, 0.07]).set_height(8.92),
            randy.animate.shift(10 * DOWN),
            run_time=2,
        )
        self.wait(8)

    def show_propagation(self, rod, run_time=10):
        rings = ProbagatingRings(rod, start_width=5)
        self.add(rings)
        self.wait(run_time)
        self.play(VFadeOut(rings))

    def get_observer(self, location=ORIGIN):
        randy = Randolph(mode="pondering")
        randy.look(RIGHT)
        randy.rotate(PI / 2, RIGHT)
        randy.rotate(PI / 2, OUT)
        randy.move_to(location)
        return randy

    def get_line_of_sign(self, rod, observer, stroke_color=WHITE, stroke_width=1):
        line = Line(ORIGIN, 5 * RIGHT)
        line.set_stroke(stroke_color, stroke_width)
        line.add_updater(lambda l: l.put_start_and_end_on(
            observer.eyes.get_top(), rod.get_center()
        ))
        line.set_flat_stroke(False)
        return line


class VectorFieldWigglingNew(InteractiveScene):
    default_frame_orientation = (-33, 85)

    def construct(self):
        # Waves
        axes, plane = get_axes_and_plane()
        self.add(axes, plane)

        wave = OscillatingWave(
            axes,
            wave_len=3.0,
            speed=1.5,
            color=BLUE,
            z_amplitude=0.5,
        )
        vector_wave = OscillatingFieldWave(axes, wave)
        wave_opacity_tracker = ValueTracker(0)
        vector_opacity_tracker = ValueTracker(1)
        wave.add_updater(lambda m: m.set_stroke(opacity=wave_opacity_tracker.get_value()))
        vector_wave.add_updater(lambda m: m.set_stroke(opacity=vector_opacity_tracker.get_value()))

        self.add(wave, vector_wave)

        # Charges
        charges = DotCloud(color=RED)
        charges.to_grid(50, 50)
        charges.set_radius(0.04)
        charges.set_height(2 * axes.z_axis.get_length())
        charges.rotate(PI / 2, RIGHT).rotate(PI / 2, IN)
        charges.move_to(axes.c2p(-10, 0, 0))
        charges.make_3d()

        charge_opacity_tracker = ValueTracker(1)
        charges.add_updater(lambda m: m.set_opacity(charge_opacity_tracker.get_value()))
        charges.add_updater(lambda m: m.set_z(0.3 * wave.xt_to_point(0, self.time)[2]))

        self.add(charges, wave, vector_wave)

        # Pan camera
        self.frame.reorient(47, 69, 0).move_to([-8.68, -7.06, 2.29]).set_height(5.44)
        self.play(
            self.frame.animate.reorient(-33, 83, 0).move_to([-0.75, -1.84, 0.38]).set_height(8.00),
            run_time=10,
        )
        self.play(
            self.frame.animate.reorient(-27, 80, 0).move_to([-0.09, -0.42, -0.1]).set_height(9.03),
            wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            run_time=4,
        )

        # Highlight x_axis
        x_line = Line(*axes.x_axis.get_start_and_end())
        x_line.set_stroke(BLUE, 10)

        self.play(
            wave_opacity_tracker.animate.set_value(0.25),
            vector_opacity_tracker.animate.set_value(0.25),
            charge_opacity_tracker.animate.set_value(0.25),
        )
        self.play(
            ShowCreation(x_line, run_time=2),
        )
        self.wait(5)

        # Show 3d wave
        wave_3d = VGroup()
        origin = axes.get_origin()
        for y in np.linspace(-1, 1, 5):
            for z in np.linspace(-1, 1, 5):
                vects = OscillatingFieldWave(
                    axes, wave,
                    max_vect_len=0.5,
                    norm_to_opacity_func=lambda n: 0.75 * np.arctan(n),
                )
                vects.y = y
                vects.z = z
                vects.add_updater(lambda m: m.shift(axes.c2p(0, m.y, m.z) - origin))
                wave_3d.add(vects)

        self.wait(2)
        wave_opacity_tracker.set_value(0)
        self.remove(vector_wave)
        self.remove(x_line)
        self.add(wave_3d)
        self.wait(2)

        self.play(
            self.frame.animate.reorient(22, 69, 0).move_to([0.41, -0.67, -0.1]).set_height(10.31),
            run_time=8
        )
        self.play(
            self.frame.animate.reorient(-48, 68, 0).move_to([0.41, -0.67, -0.1]),
            run_time=10
        )



class ClockwiseCircularLight(InteractiveScene):
    clockwise = True
    default_frame_orientation = (-20, 70)
    color = YELLOW
    x_range = (0, 24)
    amplitude = 0.5

    def setup(self):
        super().setup()
        # Axes
        axes, plane = get_axes_and_plane(x_range=self.x_range)
        self.add(axes, plane)

        # Wave
        wave = OscillatingWave(
            axes,
            wave_len=3,
            speed=0.5,
            z_amplitude=self.amplitude,
            y_amplitude=self.amplitude,
            y_phase=-PI / 2 if self.clockwise else PI / 2,
            color=self.color,
        )
        vect_wave = OscillatingFieldWave(axes, wave)
        vect_wave.set_stroke(opacity=0.7)

        self.add(wave, vect_wave)

    def construct(self):
        self.play(
            self.frame.animate.reorient(73, 82, 0),
            run_time=5
        )
        for pair in [(100, 70), (59, 72), (110, 65), (60, 80)]:
            self.play(
                self.frame.animate.reorient(*pair),
                run_time=12,
            )


class CounterclockwiseCircularLight(ClockwiseCircularLight):
    clockwise = False
    color = RED


class AltClockwiseCircularLight(ClockwiseCircularLight):
    x_range = (0, 8)
    amplitude = 0.4

    def construct(self):
        self.frame.reorient(69, 81, 0)
        self.wait(12)


class AltCounterclockwiseCircularLight(CounterclockwiseCircularLight):
    x_range = (0, 8)
    amplitude = 0.4

    def construct(self):
        self.frame.reorient(69, 81, 0)
        self.wait(12)


class TransitionTo2D(InteractiveScene):
    default_frame_orientation = (-20, 70)
    wave_config = dict(
        color=BLUE,
        wave_len=3,
    )

    def construct(self):
        # Axes
        axes, plane = get_axes_and_plane(x_range=(0, 8.01))
        self.add(axes, plane)

        # Waves
        wave = OscillatingWave(axes, **self.wave_config)
        vect_wave = OscillatingFieldWave(axes, wave)
        wave_opacity_tracker = ValueTracker(1)
        wave.add_updater(lambda m: m.set_stroke(opacity=wave_opacity_tracker.get_value()))
        vect_wave.add_updater(lambda m: m.set_stroke(opacity=wave_opacity_tracker.get_value()))

        single_vect = Vector(OUT, stroke_color=wave.get_color(), stroke_width=4)
        single_vect.set_flat_stroke(False)

        def update_vect(vect):
            x_max = axes.x_axis.x_max
            base = axes.c2p(x_max, 0, 0)
            vect.put_start_and_end_on(base, base + wave.xt_to_point(x_max, wave.time) * [0, 1, 1])

        single_vect.add_updater(update_vect)

        self.add(wave, vect_wave)

        # Shift perspective
        self.frame.reorient(69, 81, 0)
        wave_opacity_tracker.set_value(0.8)
        self.wait(6)
        self.play(
            self.frame.animate.reorient(73, 84, 0),
            VFadeIn(single_vect, time_span=(2, 3)),
            wave_opacity_tracker.animate.set_value(0.35).set_anim_args(time_span=(2, 3)),
            run_time=5,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(90, 90, 0),
            wave_opacity_tracker.animate.set_value(0),
            run_time=4,
        )
        self.wait(4)


class TransitionTo2DRightHanded(TransitionTo2D):
    wave_config = dict(
        wave_len=3,
        y_amplitude=0.5,
        z_amplitude=0.5,
        y_phase=PI / 2,
        color=RED,
    )


class TransitionTo2DLeftHanded(TransitionTo2D):
    wave_config = dict(
        wave_len=3,
        y_amplitude=0.5,
        z_amplitude=0.5,
        y_phase=-PI / 2,
        color=YELLOW,
    )


class LinearAsASuperpositionOfCircular(InteractiveScene):
    rotation_rate = 0.25
    amplitude = 2.0

    def construct(self):
        # Set up planes
        plane_config = dict(
            background_line_style=dict(stroke_color=GREY, stroke_width=1),
            faded_line_style=dict(stroke_color=GREY, stroke_width=0.5, stroke_opacity=0.5),
        )
        planes = VGroup(
            ComplexPlane((-1, 1), (-1, 1), **plane_config),
            ComplexPlane((-1, 1), (-1, 1), **plane_config),
            ComplexPlane((-2, 2), (-2, 2), **plane_config),
        )
        planes[:2].arrange(DOWN, buff=2.0).set_height(FRAME_HEIGHT - 1.5).next_to(ORIGIN, LEFT, 1.0)
        planes[2].set_height(6).next_to(ORIGIN, RIGHT, 1.0)
        # planes.arrange(RIGHT, buff=1.5)
        self.add(planes)

        # Set up trackers
        phase_trackers = ValueTracker(0).replicate(2)
        phase1_tracker, phase2_tracker = phase_trackers

        def update_phase(m, dt):
            m.increment_value(TAU * self.rotation_rate * dt)

        def slow_changer(m, dt):
            m.increment_value(-0.5 * TAU * self.rotation_rate * dt)

        for tracker in phase_trackers:
            tracker.add_updater(update_phase)

        self.add(*phase_trackers)

        def get_z1():
            return 0.5 * self.amplitude * np.exp((PI / 2 + phase1_tracker.get_value()) * 1j)

        def get_z2():
            return 0.5 * self.amplitude * np.exp((PI / 2 - phase2_tracker.get_value()) * 1j)

        def get_sum():
            return get_z1() + get_z2()

        # Vectors
        vects = VGroup(
            self.get_vector(planes[0], get_z1, color=RED),
            self.get_vector(planes[1], get_z2, color=YELLOW),
            self.get_vector(planes[2], get_sum, color=BLUE),
            self.get_vector(planes[2], get_z1, color=RED),
            self.get_vector(planes[2], get_sum, get_base=get_z1, color=YELLOW),
        )

        self.add(*vects)

        # Polarization line
        pol_line = Line(UP, DOWN)
        pol_line.set_stroke(YELLOW, 1)
        pol_line.match_height(planes[2])
        pol_line.move_to(planes[2])

        def update_pol_line(line):
            if abs(vects[2].get_length()) > 1e-3:
                line.set_angle(vects[2].get_angle())
                line.move_to(planes[2].n2p(0))
            return line

        pol_line.add_updater(update_pol_line)

        self.add(pol_line, *planes, *vects)

        # Write it as an equation
        plus = Tex("+", font_size=72)
        equals = Tex("=", font_size=72)
        plus.move_to(planes[0:2])
        # equals.move_to(planes[1:3])
        equals.move_to(ORIGIN)

        self.add(plus, equals)

        # Slow down annotation
        arcs = VGroup(
            Arrow(LEFT, RIGHT, path_arc=-PI, stroke_width=2),
            Arrow(RIGHT, LEFT, path_arc=-PI, stroke_width=2),
        )
        arcs.move_to(planes[0])
        slow_word = Text("Slow down!")
        slow_word.next_to(planes[0], DOWN)
        sucrose = Sucrose(height=1)
        sucrose.balls.scale_radii(0.25)
        sucrose.fade(0.5)
        sucrose.move_to(planes[0])
        slow_group = Group(slow_word, arcs, sucrose)

        def slow_down():
            self.play(FadeIn(slow_group, run_time=0.25))
            phase1_tracker.add_updater(slow_changer)
            self.wait(0.75)
            phase1_tracker.remove_updater(slow_changer)
            self.play(FadeOut(slow_group))

        # Highlight constituent parts
        back_rects = VGroup(*(BackgroundRectangle(plane) for plane in planes))
        back_rects.set_fill(opacity=0.5)

        self.wait(8)

        self.add(back_rects[1])
        VGroup(vects[1], vects[2], vects[4]).set_stroke(opacity=0.25)
        self.wait(8)
        self.remove(back_rects[1])

        self.add(back_rects[0])
        vects.set_stroke(opacity=1)
        VGroup(vects[0], vects[2]).set_stroke(opacity=0.25)
        self.wait(8)
        vects.set_stroke(opacity=1)
        self.remove(back_rects)
        self.wait(4)

        # Rotation labels
        for tracker in phase_trackers:
            tracker.set_value(0)

        rot_labels = VGroup(*(
            TexText("Total rotation: 0.00")
            for _ in range(2)
        ))
        for rot_label, plane, tracker in zip(rot_labels, planes, phase_trackers):
            rot_label.set_height(0.2)
            rot_label.set_color(GREY_B)
            rot_label.next_to(plane, UP)
            dec = rot_label.make_number_changable("0.00", edge_to_fix=LEFT)
            dec.phase_tracker = tracker
            dec.add_updater(lambda m: m.set_value(m.phase_tracker.get_value() / TAU))

        self.add(rot_labels)

        # Let it play, occasionally kink
        self.wait(9)
        for _ in range(20):
            slow_down()
            self.wait(3 * random.random())

    def get_vector(self, plane, get_z, get_base=lambda: 0, color=BLUE):
        vect = Vector(UP, stroke_color=color)
        vect.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(get_base()),
            plane.n2p(get_z())
        ))
        return vect




class LinearPolarization(InteractiveScene):
    default_frame_orientation = (-33, 85)

    def construct(self):
        # Waves
        axes, plane = get_axes_and_plane()
        self.add(axes, plane)

        te_wave = OscillatingWave(
            axes,
            wave_len=3.0,
            speed=1,
            color=BLUE,
            y_amplitude=0.5,
            z_amplitude=0,
        )
        te_vector_wave = OscillatingFieldWave(axes, te_wave)
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(1)
        te_wave.add_updater(lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value()))
        te_vector_wave.add_updater(lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value()))

        self.add(te_wave, te_vector_wave)

        tm_wave = OscillatingWave(
            axes,
            wave_len=3.0,
            speed=1,
            color=RED,
            z_amplitude=0.5,
        )
        tm_vector_wave = OscillatingFieldWave(axes, tm_wave)
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(1)
        tm_wave.add_updater(lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value()))
        tm_vector_wave.add_updater(lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value()))

        self.add(tm_wave, tm_vector_wave)

        pol_wave = OscillatingWave(
            axes,
            wave_len=3.0,
            speed=1,
            color=PURPLE,
            y_amplitude=0.5,
            z_amplitude=0.5,
        )
        pol_vector_wave = OscillatingFieldWave(axes, pol_wave)
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        pol_wave.add_updater(lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value()))
        pol_vector_wave.add_updater(lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value()))

        self.add(pol_wave, pol_vector_wave)

        # # Pan camera
        # # self.frame.reorient(47, 69, 0).move_to([-8.68, -7.06, 2.29]).set_height(5.44)
        # self.frame.reorient(-33, 83, 0).move_to([-0.75, -1.84, 0.38]).set_height(8.00),
        # self.play(
        #     self.frame.animate.reorient(-27, 80, 0).move_to([-0.09, -0.42, -0.1]).set_height(9.03),
        #     # wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
        #     # vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
        #     run_time=4,
        # )

        # # Highlight x_axis
        # x_line = Line(*axes.x_axis.get_start_and_end())
        # x_line.set_stroke(BLUE, 10)

        # # self.play(
        # #     wave_opacity_tracker.animate.set_value(0.25),
        # #     vector_opacity_tracker.animate.set_value(0.25),
        # # )
        # self.play(
        #     ShowCreation(x_line, run_time=2),
        # )
        # self.wait(2)

        # # Show 3d wave
        # wave_3d = VGroup()
        # origin = axes.get_origin()
        # for y in np.linspace(-1, 1, 5):
        #     for z in np.linspace(-1, 1, 5):
        #         vects = OscillatingFieldWave(
        #             axes, te_wave,
        #             max_vect_len=0.5,
        #             norm_to_opacity_func=lambda n: 0.75 * np.arctan(n),
        #         )
        #         vects.y = y
        #         vects.z = z
        #         vects.add_updater(lambda m: m.shift(axes.c2p(0, m.y, m.z) - origin))
        #         wave_3d.add(vects)

        # # self.wait(2)
        # # wave_opacity_tracker.set_value(0)
        # # self.remove(vector_wave)
        # # self.remove(x_line)
        # self.add(wave_3d)
        # self.wait(2)

        self.play(
            self.frame.animate.reorient(22, 69, 0).move_to([0.41, -0.67, -0.1]).set_height(10.31),
            # te_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            te_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # tm_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            tm_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            pol_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            # pol_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            run_time=3
        )
        self.play(
            self.frame.animate.reorient(-67, 68, 0).move_to([0.41, -0.67, -0.1]),
            run_time=3
        )


class CircularPolarization(InteractiveScene):
    default_frame_orientation = (-33, 85)

    def construct(self):
        # Waves
        axes, plane = get_axes_and_plane(
            x_range=(0, 10),
        )
        self.add(axes, plane)
        speed = 1
        wave_len = 3.0

        te_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=BLUE,
            y_amplitude=0.5,
            z_amplitude=0,
            y_phase=PI / 2,
        )
        te_vector_wave = OscillatingFieldWave(axes, te_wave)
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(1)
        te_wave.add_updater(lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value()))
        te_vector_wave.add_updater(lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value()))

        self.add(te_wave, te_vector_wave)

        tm_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=RED,
            y_amplitude=0,
            z_amplitude=0.5,
        )
        tm_vector_wave = OscillatingFieldWave(axes, tm_wave)
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(1)
        tm_wave.add_updater(lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value()))
        tm_vector_wave.add_updater(lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value()))

        self.add(tm_wave, tm_vector_wave)

        pol_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=PURPLE,
            y_amplitude=0.5,
            z_amplitude=0.5,
            y_phase=PI / 2,
        )
        pol_vector_wave = OscillatingFieldWave(axes, pol_wave)
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        pol_wave.add_updater(lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value()))
        pol_vector_wave.add_updater(lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value()))

        self.add(pol_wave, pol_vector_wave)

        self.play(
            self.frame.animate.reorient(22, 69, 0).move_to([0.41, -0.67, -0.1]).set_height(10.31),
            # te_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            te_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # tm_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            tm_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            pol_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            # pol_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            run_time=3
        )
        self.play(
            self.frame.animate.reorient(-67, 68, 0).move_to([0.41, -0.67, -0.1]),
            run_time=3
        )



class CircularPolarizationOneVec(InteractiveScene):
    default_frame_orientation = (-33, 85)

    def construct(self):
        # Waves
        axes, plane = get_axes_and_plane(
            x_range=(0, 6),
        )
        self.add(axes, plane)
        speed = 1
        wave_len = 3.0
        sample_resolution = 1
        x_density = 20
        self.y_phase_tracker = ValueTracker(0)
        self.z_phase_tracker = ValueTracker(0)

        def update_y_phase(m):
            m.y_phase = self.y_phase_tracker.get_value()

        def update_z_phase(m):
            m.z_phase = self.z_phase_tracker.get_value()

        self.te_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=BLUE,
            y_amplitude=0.5,
            z_amplitude=0,
            # y_phase=PI / 2,
            sample_resolution=sample_resolution,
        )
        te_vector_wave = OscillatingFieldWave(axes, self.te_wave,
            x_density=x_density,
        )
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(0.8)
        self.te_wave.add_updater(lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value()))
        te_vector_wave.add_updater(lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value()))

        self.add(self.te_wave, te_vector_wave)

        self.tm_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=RED,
            y_amplitude=0,
            z_amplitude=0.5,
            sample_resolution=sample_resolution,
        )
        tm_vector_wave = OscillatingFieldWave(axes, self.tm_wave,
            x_density=x_density,
        )
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(0.8)
        self.tm_wave.add_updater(lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value()))
        tm_vector_wave.add_updater(lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value()))

        self.add(self.tm_wave, tm_vector_wave)

        self.pol_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=PURPLE,
            y_amplitude=0.5,
            z_amplitude=0.5,
            # y_phase=PI / 2,
            sample_resolution=sample_resolution,
        )
        pol_vector_wave = OscillatingFieldWave(axes, self.pol_wave,
            x_density=x_density,
        )
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        self.pol_wave.add_updater(lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value()))
        pol_vector_wave.add_updater(lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value()))

        self.te_wave.add_updater(update_y_phase)
        self.tm_wave.add_updater(update_z_phase)
        self.pol_wave.add_updater(update_y_phase)
        self.pol_wave.add_updater(update_z_phase)
        self.add(self.pol_wave, pol_vector_wave)

        # self.play(
        #     self.frame.animate.reorient(22, 69, 0).move_to([0.41, -0.67, -0.1]).set_height(10.31),
        #     # te_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
        #     te_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
        #     # tm_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
        #     tm_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
        #     # pol_wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
        #     # pol_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
        #     run_time=2
        # )
        # self.play(
        #     self.frame.animate.reorient(-67, 68, 0).move_to([0.41, -0.67, -0.1]),
        #     run_time=2
        # )
        self.play(
            # self.y_phase_tracker.animate.set_value(PI).set_anim_args(time_span=(1, 2)),
            self.y_phase_tracker.animate.set_value(PI / 2),
            # te_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # tm_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # pol_wave_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            pol_vector_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            run_time=6
        )
        self.te_wave.stop_clock()
        self.tm_wave.stop_clock()
        self.pol_wave.stop_clock()

    # Key actions
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        char = chr(symbol)
        if char == "p":
            self.te_wave.toggle_clock()
            self.tm_wave.toggle_clock()
            self.pol_wave.toggle_clock()
        if char == "y":
            self.play(
                self.y_phase_tracker.animate.increment_value(PI / 8),
            )
        if char == "z":
            self.play(
                self.z_phase_tracker.animate.increment_value(PI / 8),
            )
        super().on_key_press(symbol, modifiers)

