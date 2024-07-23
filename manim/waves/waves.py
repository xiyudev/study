from manim import *
from manim.opengl import *
import numpy as np
from manim.mobject.opengl.opengl_geometry import OpenGLArrow


class WavesTest(VectorScene):
    """To generate the image/video:

    manim -qm -p --renderer=opengl --disable_caching waves.py WavesTest
    manim -qm -p --renderer=opengl --disable_caching waves.py WavesTest --write_to_movie
    """

    def construct(self):
        z_min = -5
        z_max = 5
        num_vecs = 50
        wave_len = 1
        wave_vecs = VGroup(
            *[
                Arrow3D(
                    start=[0, 0, p], end=[np.cos(p * wave_len), 0, p], color=BLUE
                )
                for p in np.linspace(z_min, z_max, num_vecs)
            ]
        )
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-5, 5],
            x_length=6,
            y_length=6,
            z_length=10,
        )

        time = ValueTracker(0)
        def advance_t(mob):
            new_end = [np.cos(mob.end[2] * wave_len + time.get_value()), 0, mob.end[2]]
            mob.become(mob.start, new_end)

        for vec in wave_vecs:
            vec.add_updater(advance_t)

        self.add(axes)
        self.add(wave_vecs)
        self.play(
            self.camera.animate.set_euler_angles(
                theta=-10 * DEGREES, phi=50 * DEGREES, gamma=80 * DEGREES
            )
        )
        self.wait()
        self.play(time.animate.set_value(5), run_time=3)
        self.wait()
        self.interactive_embed()
