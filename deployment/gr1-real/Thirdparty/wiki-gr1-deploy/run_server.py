import typer

from robot_rcs_gr.sdk.server import RobotServer


def main(
    config: str = typer.Argument(help="Path to the config file"),
    urdf_path: str = typer.Option("./urdf", help="Path to the urdf file"),
    freq: int = typer.Option(400, help="Main loop frequency in hz. defaults to 400hz."),
    debug_interval: int = typer.Option(0, help="Debug loop print interval"),
    verbose: bool = typer.Option(True, help="Print internal debug info"),
    visualize: bool = typer.Option(False, help="Visualize the robot in rviz"),
):
    if not verbose:
        from robot_rcs.logger.fi_logger import Logger

        Logger().state = Logger().STATE_OFF

    robot = RobotServer(
        config, urdf_path=urdf_path, freq=freq, debug_print_interval=debug_interval, visualize=visualize
    )
    # robot.spin()


if __name__ == "__main__":
    typer.run(main)
