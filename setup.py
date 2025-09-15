from setuptools import setup, find_packages

setup(
    name="fifteen-game",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pygame>=2.5.0",
        "numpy",
    ],
    package_data={
        "fifteen_game": ["audio/*.wav"],
    },
    entry_points={
        "console_scripts": [
            "fifteen-console=fifteen_game.console_controller:main",
            "fifteen-gui=fifteen_game.gui_controller:main",
        ],
    },
    author="Jack Draak",
    description="A sliding-tile puzzle game with GUI and console interfaces",
    keywords="game, puzzle, sliding tiles",
    python_requires=">=3.7",
)