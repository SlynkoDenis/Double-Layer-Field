import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.integrate as integrate
import scipy.special as special
from tqdm import tqdm


class Calculator:
    def __init__(self, ro1: float, ro2: float, h1: float, current: float = 1.0):
        assert 0.1 < ro1 and ro1 < 1000
        assert 0.1 < ro2 and ro2 < 1000
        assert 1 <= h1 and h1 < 100

        self.ro1 = ro1
        self.ro2 = ro2
        self.h1 = h1
        self.current = current

    def contrast_coefficient(self) -> float:
        return (self.ro2 - self.ro1) / (self.ro2 + self.ro1)

    def __str__(self) -> str:
        return f'EM Field: ro1 = {self.ro1}, ro2 = {self.ro2}, h1 = {self.h1}'

    def __call__(self, radius: float) -> float:
        assert 0.1 <= radius and radius <= 10_000
        return self.ro1 * self.current / 2 / np.pi *\
            (1 / radius + 2 * self.contrast_coefficient() * self.__calculate_integral(radius))

    def __calculate_integral(self, radius: float) -> float:
        return integrate.quad(self.__get_integrate_function(radius), 0, float('inf'))[0]

    def __get_integrate_function(self, radius: float):
        def foo(m: float) -> float:
            exp = np.exp(-2 * m * self.h1)
            return exp / (1 - self.contrast_coefficient() * exp) * special.j0(m * radius)
        return foo


def calculate(calc: Calculator) -> tuple[np.ndarray, np.ndarray]:
    rng = np.arange(0.1, 10_000, step=1)
    return rng, np.array([calc(x) for x in tqdm(rng)])


def plot_interactive(x: np.ndarray, y: np.ndarray, plot_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(dash='solid')))

    fig.update_yaxes(title_text='Voltage, V')
    fig.update_xaxes(title_text='Spacing, m')
    fig.update_layout(height=800, width=1500)
    fig.update_layout(hoverlabel_namelength=-1)

    fig.write_html(f'{plot_name}.html')


def plot_static(x: np.ndarray, y: np.ndarray, title: str, plot_name: str):
    fig = plt.figure(figsize=[16, 12])

    plt.plot(x, y)
    plt.grid()
    plt.title(title)
    plt.xlabel('Spacing, m')
    plt.ylabel('Voltage, V')
    plt.tight_layout()

    fig.savefig(f'{plot_name}.png')


def main(ro1: float, ro2: float, h1: float, plot_name: str):
    calc = Calculator(ro1, ro2, h1)
    x, y = calculate(calc)
    plot_static(x, y, str(calc), plot_name)
    plot_interactive(x, y, plot_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='EM Field Plotter')
    parser.add_argument('--name', type=str, default='plot', required=False)
    parser.add_argument('--ro1', type=float, default=1, required=False)
    parser.add_argument('--ro2', type=float, default=1, required=False)
    parser.add_argument('--h1', type=float, default=1, required=False)
    args = parser.parse_args()

    main(args.ro1, args.ro2, args.h1, args.name)
