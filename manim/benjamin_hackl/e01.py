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
