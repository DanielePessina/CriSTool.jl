# Process conditions

```@docs
AbstractTemperature
ConstantTemperature
LinearTemperature
RampTemperature
CallableTemperature
temperature
AbstractSolubilityModel
AbstractSaturationModel
ConstantSolubility
PolynomialSolubility
CallableSolubility
lysozyme_solubility
lysozyme_saturation
saturation_concentration
supersaturation
```

`AbstractSaturationModel` is a compatibility alias for
`AbstractSolubilityModel`. `ConstantSaturation`, `PolynomialSaturation`, and
`CallableSaturation` are aliases for the corresponding `*Solubility`
constructors. `lysozyme_saturation()` is the corresponding alias for
`lysozyme_solubility()`.
