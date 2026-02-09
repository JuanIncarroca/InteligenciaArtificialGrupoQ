def calcular_imc_tradicional(peso, altura):
    # La regla es inamovible
    imc = peso / (altura ** 2)
    return round(imc, 2)

# Ejemplo: 75kg y 1.75m
resultado = calcular_imc_tradicional(75, 1.75)
print(f"IMC Tradicional: {resultado}") # 24.49
