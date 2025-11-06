"""Module entry point for `python -m soundstream_light`."""

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

