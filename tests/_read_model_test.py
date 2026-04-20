def test_model_data(solwin_model):
    xflux = solwin_model['xflux']
    yflux = solwin_model['yflux']
    zflux = solwin_model['zflux']
    flxbin = solwin_model['flxbin']
    numbin = solwin_model['numbin']
    numdat = solwin_model['numdat']

    #: Test that the data is loaded correctly
    assert len(xflux) == 9  #: MAXKP = 9
    assert len(yflux) == 9
    assert len(zflux) == 9
    assert len(flxbin) == 9
    assert len(numbin) == 9
    assert len(numdat) == 9

    #: Test that the first KP level has the expected number of data points
    assert numdat[0] > 0

    #: Test that the flux values are within a reasonable range (example check)
    for k in range(9):
        for f in flxbin[k]:
            assert f >= 0.0  #: Flux should be non-negative

def test_index_offset(index_offset):
    nsphvol3 = index_offset['nsphvol3']
    ioffset3 = index_offset['ioffset3']
    joffset3 = index_offset['joffset3']
    koffset3 = index_offset['koffset3']

    assert ioffset3.dtype == int
    assert joffset3.dtype == int
    assert koffset3.dtype == int
    assert len(ioffset3) == nsphvol3
    assert len(joffset3) == nsphvol3
    assert len(koffset3) == nsphvol3

def test_solar_wind_parameters(solar_wind_parameters):
    
    assert len(solar_wind_parameters) == 28
    assert isinstance(solar_wind_parameters, dict)