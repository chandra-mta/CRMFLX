# Model Data Documentation


## Files

### index_offset.json

**Derivation:** derive_index_offset.ipynb

```
distmapmax3 = 20.0
xinc    = 1.0       #--- Chandra Volume Element Database Parameters: length of volume element in x-direction (Re)
yinc    = 1.0       #--- Chandra Volume Element Database Parameters: length of volume element in y-direction (Re)
zinc    = 1.0       #--- Chandra Volume Element Database Parameters: length of volume element in z-direction (Re)
maxnsphvol = 100000 #--- maximum number of sub-volume elements stores in the streamline mapping search volume
```
#### Data Values Description
- **nsphvol:** number of volume elements stored in the database search volume.
- **ioffset:** array of offset indices for x-direction.
- **joffset:** array of offset indices for y-direction.
- **koffset:** array of offset indices for z-direction.

### solar_wind_parameters.json

**Derivation:** derive_solar_wind_parameters.ipynb

Stores the KP-dependent solar wind parameters used as inputs for the bow shock and magnetopause boundary models.

#### Data Values Description

##### Key
- **key_xkp:** String-formatted KP Index value with two-decimal places
```
key_xkp = f"{xkp:.2f}"
```

##### Values
- **bx:** the imf b_x (nt)
- **by:** the imf b_y (nt)
- **bz:** the imf b_z (nt)
- **vx:** x component of solar wind bulk flow velocity (km/s).
- **vy:** y component of solar wind bulk flow velocity (km/s).
- **vz:** z component of solar wind bulk flow velocity (km/s).
- **dennum:** the solar wind proton number density (#/cm^3)
- **swetemp:** the solar wind electron temperature (k)
- **swptemp:** the solar wind proton temperature (k)
- **hefrac:** fraction of solar wind ions which are helium ions
- **swhtemp:** the temperature of the helium (k)
- **bowang:** angle bow shock radius calculated (rad).
- **dypres**: solar wind dynamic pressure (np).
- **abang:** aberration angle of magnetotail (deg).
- **xhinge:** hinge point of magnetotail (re).