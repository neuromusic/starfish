from typing import Collection, Mapping, Tuple

import numpy as np

from starfish.types import Coordinates, Indices, Number
from ._key import TileKey


class TileData:
    """
    Base class for a parser to implement that provides the data for a single tile.
    """
    @property
    def tile_shape(self) -> Tuple[int, int]:
        raise NotImplementedError()

    @property
    def numpy_array(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def coordinates(self) -> Mapping[Coordinates, Tuple[Number, Number]]:
        raise NotImplementedError()

    @property
    def indices(self) -> Mapping[Indices, int]:
        raise NotImplementedError()


class TileCollectionData:
    """
    Base class for a parser to implement that provides the data for a collection of tiles to be
    assembled into an ImageStack.
    """
    def __getitem__(self, tilekey: TileKey) -> dict:
        """Returns the extras metadata for a given tile, addressed by its TileKey"""
        raise NotImplementedError()

    def keys(self) -> Collection[TileKey]:
        """Returns a Collection of the TileKey's for all the tiles."""
        raise NotImplementedError()

    @property
    def extras(self) -> dict:
        """Returns the extras metadata for the TileSet."""
        raise NotImplementedError()

    def get_tile_by_key(self, tilekey: TileKey) -> TileData:
        raise NotImplementedError()

    def get_tile(self, r: int, ch: int, z: int) -> TileData:
        raise NotImplementedError()
