from __future__ import annotations

from manimlib import *
from matplotlib import colormaps

spectral_cmap = colormaps.get_cmap("Spectral")

# Helper functions


def get_spectral_color(alpha):
    return Color(rgb=spectral_cmap(alpha)[:3])


def get_axes_and_plane(
    x_range=(0, 24),
    y_range=(-1, 1),
    z_range=(-1, 1),
    x_unit=1,
    y_unit=2,
    z_unit=2,
    origin_point=2 * LEFT,
    axes_opacity=0.5,
    plane_line_style=dict(stroke_color=GREY_C, stroke_width=1, stroke_opacity=0.5),
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
        axes.x_range,
        axes.y_range,
        width=axes.x_axis.get_length(),
        height=axes.y_axis.get_length(),
        background_line_style=plane_line_style,
        axis_config=dict(stroke_width=0),
    )
    plane.shift(axes.get_origin() - plane.get_origin())
    plane.set_flat_stroke(False)

    return axes, plane


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
        xs = np.arange(self.axes.x_axis.x_min, self.axes.x_axis.x_max, self.sample_resolution)
        self.set_points_as_corners(self.offset + self.xt_to_point(xs, self.time))

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
        return get_spectral_color(inverse_interpolate(1.5, 0.5, wave_len))


class OscillatingWaveSum(OscillatingWave):
    def __init__(
        self,
        axes,
        waves,
        **kwargs,
    ):
        self.axes = axes
        self.waves = waves

        super().__init__(axes, **kwargs)

    def xt_to_yz(self, x, t):
        y_sum, z_sum = 0, 0
        for wave in self.waves:
            y, z = wave.xt_to_yz(x, t)
            y_sum += y
            z_sum += z
        return y_sum, z_sum


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
        **kwargs,
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
            center, width, height, depth, x_density, y_density, z_density
        )
        self.init_base_stroke_width_array(len(self.sample_points))

        super().__init__(stroke_color=stroke_color, flat_stroke=flat_stroke, **kwargs)

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
        z_density: float,
    ) -> np.ndarray:
        to_corner = np.array([width / 2, height / 2, depth / 2])
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        to_corner = spacings * (to_corner / spacings).astype(int)
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        # print(f"XIYU DEBUG {center=}, {width=}, {height=}, {depth=}, {x_density=}, {y_density=}, {z_density=}")
        # print(f"XIYU DEBUG {to_corner=}, {spacings=}, {lower_corner=}, {upper_corner=}")
        return cartesian_product(
            *(
                np.arange(low, high, space)
                for low, high, space in zip(lower_corner, upper_corner, spacings)
            )
        )

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
            points[i::8] = 0.5 * (points[i - 1 :: 8] + points[i + 1 :: 8])
        points[7::8] = points[6:-1:8]

        # Adjust stroke widths
        width_arr = self.stroke_width * self.base_stroke_width_array
        width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        width_scalars = np.repeat(width_scalars, 8)[:-1]
        self.get_stroke_widths()[:] = width_scalars * width_arr

        # Potentially adjust opacity and color
        if self.norm_to_opacity_func is not None:
            self.get_stroke_opacities()[:] = self.norm_to_opacity_func(np.repeat(norms, 8)[:-1])
        if self.norm_to_rgb_func is not None:
            self.get_stroke_colors()
            self.data["stroke_rgba"][:, :3] = self.norm_to_rgb_func(np.repeat(norms, 8)[:-1])

        self.note_changed_data()
        return self


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

        super().__init__(func=vector_func, max_vect_len=max_vect_len, **kwargs)
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
        super().__init__(axes=axes, graph_func=lambda x: wave.xt_to_yz(x, wave.time), **kwargs)

    def get_sample_points(self, *args, **kwargs):
        # Override super class and ignore all length/density information
        # print(f"XIYU DEBUG {self.sample_xs=}")
        return self.wave.offset + self.axes.x_axis.n2p(self.sample_xs)


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
        x_density = 10
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
        te_vector_wave = OscillatingFieldWave(
            axes,
            self.te_wave,
            x_density=x_density,
        )
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(0.8)
        self.te_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value())
        )
        te_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value())
        )

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
        tm_vector_wave = OscillatingFieldWave(
            axes,
            self.tm_wave,
            x_density=x_density,
        )
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(0.8)
        self.tm_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value())
        )
        tm_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value())
        )

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
        pol_vector_wave = OscillatingFieldWave(
            axes,
            self.pol_wave,
            x_density=x_density,
        )
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        self.pol_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value())
        )
        pol_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value())
        )

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
        self.wait(2)
        self.play(
            # self.y_phase_tracker.animate.set_value(PI).set_anim_args(time_span=(1, 2)),
            self.y_phase_tracker.animate.set_value(PI / 2),
            # te_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # tm_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # pol_wave_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            pol_vector_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            run_time=6,
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


class PSR(InteractiveScene):
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
        x_density = 10
        self.y_phase_tracker = ValueTracker(0)
        self.z_phase_tracker = ValueTracker(0)
        y_amplitude = 0.5
        z_amplitude = 0

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
        te_vector_wave = OscillatingFieldWave(
            axes,
            self.te_wave,
            x_density=x_density,
        )
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(0.5)
        self.te_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value())
        )
        te_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value())
        )

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
        tm_vector_wave = OscillatingFieldWave(
            axes,
            self.tm_wave,
            x_density=x_density,
        )
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(0.5)
        self.tm_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value())
        )
        tm_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value())
        )

        self.add(self.tm_wave, tm_vector_wave)

        # self.pol_wave = OscillatingWave(
        #     axes,
        #     wave_len=wave_len,
        #     speed=speed,
        #     color=PURPLE,
        #     y_amplitude=0.5,
        #     z_amplitude=0.5,
        #     # y_phase=PI / 2,
        #     sample_resolution=sample_resolution,
        # )
        self.pol_wave = OscillatingWaveSum(
            axes,
            waves=[self.te_wave, self.tm_wave],
        )
        pol_vector_wave = OscillatingFieldWave(
            axes,
            self.pol_wave,
            x_density=x_density,
        )
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        self.pol_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value())
        )
        pol_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value())
        )

        self.te_wave.add_updater(update_y_phase)
        self.tm_wave.add_updater(update_z_phase)
        # self.pol_wave.add_updater(update_y_phase)
        # self.pol_wave.add_updater(update_z_phase)
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
            run_time=6,
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


class PSRInput(InteractiveScene):
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
        x_density = 10
        amplitude = 0.5
        # The angle of the electric field vector.
        self.angle_tracker = ValueTracker(PI / 6)
        self.y_phase_tracker = ValueTracker(0)
        self.z_phase_tracker = ValueTracker(0)

        def update_y_phase(m):
            m.y_phase = self.y_phase_tracker.get_value()

        def update_z_phase(m):
            m.z_phase = self.z_phase_tracker.get_value()

        def update_y_amp(m):
            m.y_amplitude = amplitude * np.cos(self.angle_tracker.get_value())

        def update_z_amp(m):
            m.z_amplitude = amplitude * np.sin(self.angle_tracker.get_value())

        self.te_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=BLUE,
            y_amplitude=amplitude * np.cos(self.angle_tracker.get_value()),
            z_amplitude=0,
            # y_phase=PI / 2,
            sample_resolution=sample_resolution,
        )
        te_vector_wave = OscillatingFieldWave(
            axes,
            self.te_wave,
            x_density=x_density,
        )
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(0.1)
        self.te_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value())
        )
        te_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value())
        )

        self.add(self.te_wave, te_vector_wave)

        self.tm_wave = OscillatingWave(
            axes,
            wave_len=wave_len,
            speed=speed,
            color=RED,
            y_amplitude=0,
            z_amplitude=amplitude * np.sin(self.angle_tracker.get_value()),
            sample_resolution=sample_resolution,
        )
        tm_vector_wave = OscillatingFieldWave(
            axes,
            self.tm_wave,
            x_density=x_density,
        )
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(0.1)
        self.tm_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value())
        )
        tm_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value())
        )

        self.add(self.tm_wave, tm_vector_wave)

        # self.pol_wave = OscillatingWave(
        #     axes,
        #     wave_len=wave_len,
        #     speed=speed,
        #     color=PURPLE,
        #     y_amplitude=0.5,
        #     z_amplitude=0.5,
        #     # y_phase=PI / 2,
        #     sample_resolution=sample_resolution,
        # )
        self.pol_wave = OscillatingWaveSum(
            axes,
            waves=[self.te_wave, self.tm_wave],
        )
        pol_vector_wave = OscillatingFieldWave(
            axes,
            self.pol_wave,
            x_density=x_density,
        )
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        self.pol_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value())
        )
        pol_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value())
        )

        self.te_wave.add_updater(update_y_phase)
        self.tm_wave.add_updater(update_z_phase)
        self.te_wave.add_updater(update_y_amp)
        self.tm_wave.add_updater(update_z_amp)
        self.add(self.pol_wave, pol_vector_wave)

        # self.wait(3)
        self.play(
            # self.y_phase_tracker.animate.set_value(PI).set_anim_args(time_span=(1, 2)),
            # self.y_phase_tracker.animate.set_value(PI / 2),
            # te_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # tm_vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            # pol_wave_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            self.angle_tracker.animate.set_value(PI / 6 * 2),
            pol_vector_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            run_time=3,
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
        if char == "a":
            self.play(
                self.angle_tracker.animate.increment_value(PI / 12),
            )
        super().on_key_press(symbol, modifiers)


class PSRRotateTM(InteractiveScene):
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
        x_density = 10
        amplitude = 0.5
        # The angle of the electric field vector.
        self.tm_angle_tracker = ValueTracker(PI / 2)
        self.te_phase_tracker = ValueTracker(0)
        self.tm_phase_tracker = ValueTracker(0)

        def update_te_y_phase(m):
            m.y_phase = self.te_phase_tracker.get_value()

        def update_tm_y_phase(m):
            m.y_phase = self.tm_phase_tracker.get_value()

        def update_tm_z_phase(m):
            m.z_phase = self.tm_phase_tracker.get_value()

        def update_tm_y_amp(m):
            m.y_amplitude = amplitude * np.cos(self.tm_angle_tracker.get_value())

        def update_tm_z_amp(m):
            m.z_amplitude = amplitude * np.sin(self.tm_angle_tracker.get_value())

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
        te_vector_wave = OscillatingFieldWave(
            axes,
            self.te_wave,
            x_density=x_density,
        )
        te_wave_opacity_tracker = ValueTracker(0)
        te_vector_opacity_tracker = ValueTracker(0.1)
        self.te_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_wave_opacity_tracker.get_value())
        )
        te_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=te_vector_opacity_tracker.get_value())
        )

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
        tm_vector_wave = OscillatingFieldWave(
            axes,
            self.tm_wave,
            x_density=x_density,
        )
        tm_wave_opacity_tracker = ValueTracker(0)
        tm_vector_opacity_tracker = ValueTracker(0.1)
        self.tm_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_wave_opacity_tracker.get_value())
        )
        tm_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=tm_vector_opacity_tracker.get_value())
        )

        self.add(self.tm_wave, tm_vector_wave)

        # self.pol_wave = OscillatingWave(
        #     axes,
        #     wave_len=wave_len,
        #     speed=speed,
        #     color=PURPLE,
        #     y_amplitude=0.5,
        #     z_amplitude=0.5,
        #     # y_phase=PI / 2,
        #     sample_resolution=sample_resolution,
        # )
        self.pol_wave = OscillatingWaveSum(
            axes,
            waves=[self.te_wave, self.tm_wave],
        )
        pol_vector_wave = OscillatingFieldWave(
            axes,
            self.pol_wave,
            x_density=x_density,
        )
        pol_wave_opacity_tracker = ValueTracker(0)
        pol_vector_opacity_tracker = ValueTracker(1)
        self.pol_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_wave_opacity_tracker.get_value())
        )
        pol_vector_wave.add_updater(
            lambda m: m.set_stroke(opacity=pol_vector_opacity_tracker.get_value())
        )

        self.te_wave.add_updater(update_te_y_phase)
        self.tm_wave.add_updater(update_tm_y_phase)
        self.tm_wave.add_updater(update_tm_z_phase)
        self.tm_wave.add_updater(update_tm_y_amp)
        self.tm_wave.add_updater(update_tm_z_amp)
        self.add(self.pol_wave, pol_vector_wave)

        # self.wait(3)
        self.play(
            # self.te_phase_tracker.animate.set_value(PI).set_anim_args(time_span=(1, 2)),
            # self.te_phase_tracker.animate.set_value(PI / 2),
            te_vector_opacity_tracker.animate.set_value(0.3).set_anim_args(time_span=(1, 2)),
            tm_vector_opacity_tracker.animate.set_value(0.3).set_anim_args(time_span=(1, 2)),
            # pol_wave_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            # self.tm_angle_tracker.animate.set_value(PI / 6 * 2),
            pol_vector_opacity_tracker.animate.set_value(1.0).set_anim_args(time_span=(1, 2)),
            run_time=3,
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
                self.te_phase_tracker.animate.increment_value(PI / 8),
            )
        if char == "z":
            self.play(
                self.tm_phase_tracker.animate.increment_value(PI / 8),
            )
        if char == "m":
            self.play(
                self.tm_angle_tracker.animate.increment_value(-PI / 12),
            )
        super().on_key_press(symbol, modifiers)
