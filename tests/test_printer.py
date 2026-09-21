"""The machine definition."""

from __future__ import annotations

import pytest

from electrode_slicer.printer import PRINTERS, VOLTERA_NOVA, get_printer


def test_voltera_nova_matches_the_machine():
    assert VOLTERA_NOVA.plate == (220.0, 300.0, 40.0)
    assert VOLTERA_NOVA.line_width == 0.1, "the NOVA lays a 100 um bead"
    assert VOLTERA_NOVA.name in PRINTERS


def test_usable_area_excludes_the_margin_on_both_sides(printer):
    assert printer.usable_x == pytest.approx(220.0 - 2 * 10.0)
    assert printer.usable_y == pytest.approx(300.0 - 2 * 10.0)


def test_first_layer_defaults_to_the_layer_height(printer):
    assert printer.first_layer == printer.layer_height
    thicker = printer.with_overrides(first_layer_height=0.2)
    assert thicker.first_layer == 0.2
    assert thicker.layer_height == printer.layer_height


def test_overrides_ignore_none_so_unset_cli_flags_change_nothing(printer):
    same = printer.with_overrides(line_width=None, layer_height=None)
    assert same == printer
    changed = printer.with_overrides(line_width=0.4)
    assert changed.line_width == 0.4
    assert printer.line_width == 0.25, "the original must not be mutated"


@pytest.mark.parametrize(
    "bad",
    [
        {"plate_x": 0},
        {"plate_y": -1},
        {"z_travel": 0},
        {"line_width": 0},
        {"layer_height": -0.1},
        {"first_layer_height": 0},
        {"margin": -1},
        {"margin": 200},        # leaves no usable plate
    ],
)
def test_impossible_machines_are_rejected(printer, bad):
    with pytest.raises(ValueError):
        printer.with_overrides(**{k: v for k, v in bad.items()})


def test_check_part_rejects_something_taller_than_the_z_travel(printer):
    with pytest.raises(ValueError, match="Z travel"):
        printer.check_part(5.0, 5.0, 100.0)


def test_check_part_rejects_a_footprint_larger_than_the_plate(printer):
    with pytest.raises(ValueError, match="does not fit"):
        printer.check_part(500.0, 5.0, 1.0)
    printer.check_part(5.0, 5.0, 1.0)     # a normal part raises nothing


def test_unknown_printer_names_say_what_is_available():
    with pytest.raises(KeyError, match="voltera-nova"):
        get_printer("nonexistent")
    assert get_printer("voltera-nova") is VOLTERA_NOVA
