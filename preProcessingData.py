import numpy as np

nome_arquivo_entrada = "../dataset/sintetico/2d-4c-no4.dat"
nome_arquivo_saida = "../dataset/sintetico/2d-4c-no4.csv"

# Lê o arquivo de entrada e processa as linhas
linhas_processadas = []
with open(nome_arquivo_entrada, "r") as arquivo:
    for linha in arquivo:
        partes = linha.strip().split()  # Divide a linha em partes separadas por espaços
        linhas_processadas.append(";".join(partes))  # Junta as partes com ponto e vírgula

# Escreve as linhas processadas no arquivo CSV de saída
with open(nome_arquivo_saida, "w") as arquivo:
    for linha in linhas_processadas:
        arquivo.write(f"{linha}\n")

mat = np.genfromtxt("../dataset/sintetico/2d-4c-no4.csv", dtype=float, delimiter=';', missing_values=np.nan)
data = mat[::, :-1]
groundTruth = mat[::, -1]
with open("../dataset/sintetico/groundTruth.csv", "w") as arquivo:
    for item in range(groundTruth.shape[0]):
        arquivo.write(f"{groundTruth[item]}\n")

with open(nome_arquivo_saida, "w") as arquivo:
    for linha in range(data.shape[0]):
        for coluna in range(data.shape[1]):
            if(coluna == data.shape[1] - 1):
                arquivo.write(f"{data[linha,coluna]}\n")
            else:
                arquivo.write(f"{data[linha,coluna]};")