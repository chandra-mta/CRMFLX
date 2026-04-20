import pytest
from src.crmflx._read_model import load_index_offset, load_model, load_solar_wind_parameters

@pytest.fixture
def solwin_model():
    return load_model("solwin")

@pytest.fixture
def msheath_model():
    return load_model("msheath")

@pytest.fixture
def msph_model():
    return load_model("msph")

@pytest.fixture
def index_offset():
    return load_index_offset()

@pytest.fixture
def solar_wind_parameters():
    return load_solar_wind_parameters()