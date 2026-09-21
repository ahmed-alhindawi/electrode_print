"""The machine: build volume, bead geometry and what it can physically do.

Everything that describes the printer rather than the design lives here, so a
second machine is a new ``Printer`` instance and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["Printer", "VOLTERA_NOVA", "PRINTERS", "get_printer"]


@dataclass(frozen=True)
class Printer:
    """A dispensing printer.

    Lengths are millimetres.  ``line_width`` is the width of the bead the
    nozzle lays down, which sets both the toolpath spacing and the stroke
    width written into the SVG.
    """

    name: str
    plate_x: float
    plate_y: float
    z_travel: float
    line_width: float = 0.1
    layer_height: float = 0.1
    first_layer_height: float | None = None
    margin: float = 10.0
    resolution: float = 0.005
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in ("plate_x", "plate_y", "z_travel", "line_width", "layer_height"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{self.name}: {field_name} must be positive")
        if self.first_layer_height is not None and self.first_layer_height <= 0:
            raise ValueError(f"{self.name}: first_layer_height must be positive")
        if self.margin < 0:
            raise ValueError(f"{self.name}: margin cannot be negative")
        if self.margin * 2 >= min(self.plate_x, self.plate_y):
            raise ValueError(f"{self.name}: margin leaves no usable plate")

    @property
    def plate(self) -> tuple[float, float, float]:
        return (self.plate_x, self.plate_y, self.z_travel)

    @property
    def first_layer(self) -> float:
        """The first layer is often thicker to key into the substrate."""
        return self.first_layer_height if self.first_layer_height else self.layer_height

    @property
    def usable_x(self) -> float:
        return self.plate_x - 2 * self.margin

    @property
    def usable_y(self) -> float:
        return self.plate_y - 2 * self.margin

    def check_part(self, width: float, depth: float, height: float) -> None:
        """Raise if a single part cannot be printed at all."""
        if height > self.z_travel:
            raise ValueError(
                f"part is {height:.2f} mm tall, taller than the "
                f"{self.z_travel:g} mm Z travel of the {self.name}"
            )
        if width > self.usable_x or depth > self.usable_y:
            raise ValueError(
                f"part footprint {width:.2f} x {depth:.2f} mm does not fit the usable "
                f"plate area {self.usable_x:.1f} x {self.usable_y:.1f} mm of the "
                f"{self.name} (reduce the margin or scale the part down)"
            )

    def with_overrides(self, **changes) -> Printer:
        """A copy with some settings replaced, ignoring any left as None."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})


VOLTERA_NOVA = Printer(
    name="voltera-nova",
    plate_x=220.0,
    plate_y=300.0,
    z_travel=40.0,
    line_width=0.1,      # 100 um dispensed bead
    layer_height=0.1,
    margin=10.0,
    description="Voltera NOVA flexible-electronics dispensing printer",
)

PRINTERS: dict[str, Printer] = {VOLTERA_NOVA.name: VOLTERA_NOVA}


def get_printer(name: str) -> Printer:
    try:
        return PRINTERS[name]
    except KeyError:
        known = ", ".join(sorted(PRINTERS))
        raise KeyError(f"unknown printer {name!r}; known printers: {known}") from None
