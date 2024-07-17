import manim


class Positioning(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e02.py Positioning
    """

    def construct(self):
        plane = manim.NumberPlane()
        self.add(plane)

        # next_to
        red_dot = manim.Dot(color=manim.RED)
        green_dot = manim.Dot(color=manim.GREEN)
        green_dot.next_to(red_dot, manim.RIGHT + manim.UP)  # RIGHT = [1, 0 ,0]
        self.add(red_dot, green_dot)

        # shift
        s = manim.Square(color=manim.ORANGE)
        s.shift(2 * manim.UP + 4 * manim.RIGHT)
        self.add(s)

        # move_to
        c = manim.Circle(color=manim.PURPLE)
        c.move_to([-3, -2, 0])
        self.add(c)

        # align_to
        c2 = manim.Circle(radius=0.5, color=manim.RED, fill_opacity=0.5)
        c3 = c2.copy().set_color(manim.YELLOW)
        c4 = c2.copy().set_color(manim.ORANGE)
        c2.align_to(s, manim.UP)
        c3.align_to(s, manim.RIGHT)
        c4.align_to(s, manim.RIGHT + manim.UP)
        self.add(c2, c3, c4)


class CriticalPoints(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e02.py CriticalPoints
    """

    def construct(self):
        c = manim.Circle(color=manim.GREEN, fill_opacity=0.5)
        self.add(c)

        for d in [
            (0, 0, 0),
            manim.UP,
            manim.UR,
            manim.RIGHT,
            manim.DR,
            manim.DOWN,
            manim.DL,
            manim.LEFT,
            manim.UL,
        ]:
            self.add(manim.Cross(scale_factor=0.2).move_to(c.get_critical_point(d)))

        s = manim.Square(color=manim.RED, fill_opacity=0.5)
        s.move_to([1, 0, 0], aligned_edge=manim.LEFT)
        self.add(s)


class UsefulUnits(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e02.py UsefulUnits
    """

    def construct(self):
        # self.add(manim.Axes())
        for perc in range(5, 51, 5):
            self.add(manim.Circle(radius=perc * manim.unit.Percent(manim.X_AXIS)))
            self.add(
                manim.Square(
                    side_length=2 * perc * manim.unit.Percent(manim.Y_AXIS),
                    color=manim.YELLOW,
                )
            )

        d = manim.Dot()
        d.shift(100 * manim.unit.Pixels * manim.RIGHT)
        self.add(d)


class Grouping(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e02.py Grouping
    """

    def construct(self):
        red_dot = manim.Dot(color=manim.RED)
        green_dot = manim.Dot(color=manim.GREEN).next_to(red_dot, manim.RIGHT)
        blue_dot = manim.Dot(color=manim.BLUE).next_to(red_dot, manim.UP)
        dot_group = manim.VGroup(red_dot, green_dot, blue_dot)
        dot_group.to_edge(manim.RIGHT)
        self.add(dot_group)

        circles = manim.VGroup(*[manim.Circle(radius=0.2) for _ in range(10)])
        circles.arrange(manim.UP, buff=0.1)
        self.add(circles)

        # stars = manim.VGroup(
        #     *[
        #         manim.Star(color=manim.YELLOW, fill_opacity=1).scale(0.5)
        #         for _ in range(20)
        #     ]
        # )
        stars = manim.VGroup(
            *[
                manim.Star(color=manim.YELLOW, fill_opacity=0.2).scale(0.3)
                for _ in range(20)
            ]
        )
        stars.arrange_in_grid(4, 5, buff=0.5)
        self.add(stars)


# manim.config.background_color = manim.DARK_BLUE
# # The frame_width and frame_height are in munits.
# manim.config.frame_width = 16
# manim.config.frame_height = 9
# manim.config.pixel_width = 1920
# manim.config.pixel_height = 1080


class ConfigSetting(manim.Scene):
    """A simple example.

    Uncomment the manim.config.* lines above, and run:

      manim -qm -p e01.py ConfigSetting
    """

    def construct(self):
        plane = manim.NumberPlane(x_range=(-8, 8), y_range=(-4.5, 4.5))
        t = manim.Triangle(color=manim.PURPLE, fill_opacity=0.5)
        self.add(plane, t)


# manim.config.background_color = manim.WHITE


class ChangeDefaults(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e01.py ChangeDefaults
    """

    def construct(self):
        manim.Text.set_default(color=manim.GREEN, font_size=100)
        t = manim.Text("Hello World!")
        self.add(t)

        # Calling without any arguments to restore the default behavior.
        manim.Text.set_default()
        t2 = manim.Text("Goodbye!").next_to(t, manim.DOWN)
        self.add(t2)
