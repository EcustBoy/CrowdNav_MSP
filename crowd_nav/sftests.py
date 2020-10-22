import numpy as np
import socialforce


def test_crossing():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.5, 10.0, 10.0],
        [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],
    ])
    s = socialforce.Simulator(initial_state)
    states = np.stack([s.step().state.copy() for _ in range(50)])
    print(states)

    # visualize
    print('')
    with socialforce.show.canvas() as ax:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()
if __name__ == "__main__":
    test_crossing()