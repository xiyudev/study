import manim


class FirstExample(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e01.py FirstExample
    """

    def construct(self):
        blue_circle = manim.Circle(color=manim.BLUE, fill_opacity=0.5)
        green_square = manim.Square(color=manim.GREEN, fill_opacity=0.5)
        green_square.next_to(blue_circle, manim.RIGHT)
        self.add(blue_circle, green_square)


class SecondExample(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e01.py SecondExample
    """

    def construct(self):
        ax = manim.Axes(x_range=(-3, 3), y_range=(-3, 3))
        curve = ax.plot(lambda x: (x + 2) * x * (x - 2) / 2, color=manim.RED)
        area = ax.get_area(curve, x_range=(-2, 0))
        self.add(ax, curve, area)


class ThirdExample(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e01.py ThirdExample
    """

    def construct(self):
        ax = manim.Axes(x_range=(-3, 3), y_range=(-3, 3))
        curve = ax.plot(lambda x: (x + 2) * x * (x - 2) / 2, color=manim.RED)
        area = ax.get_area(curve, x_range=(-2, 0))
        # self.play(manim.Create(ax), manim.Create(curve), run_time=3)
        self.play(manim.Create(ax, run_time=3), manim.Create(curve, run_time=5))
        self.play(manim.FadeIn(area))
        self.wait(2)


class SquareToCircle(manim.Scene):
    """A simple example.

    To generate the image:

      manim -qm -p e01.py SquareToCircle
    """

    def construct(self):
        green_square = manim.Square(color=manim.GREEN, fill_opacity=0.5)
        self.play(manim.DrawBorderThenFill(green_square))
        blue_circle = manim.Circle(color=manim.BLUE, fill_opacity=0.5)
        # self.play(manim.Transform(green_square, blue_circle))
        # self.play(manim.FadeOut(green_square))
        self.play(manim.ReplacementTransform(green_square, blue_circle))
        self.play(manim.Indicate(blue_circle))
        self.play(manim.FadeOut(blue_circle))
        self.wait(2)
