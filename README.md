# Aprendizado Supervisionado para a Reconstrução de Imagens por Tomografia por Impedância Elétrica
Código para a pesquisa de iniciação científica no Departamento de Eng. Mecatrônica e Sistemas Mecânicos da Escola
Politécnica da USP, no problema de Reconstrução de Imagens por Tomografia por Impedância Elétrica,
utilizando Aprendizado Supervisionado.

O projeto consiste em utilizar uma rede neural convolucional, baseada na LeNet, para reconstruir a imagem do interior
de um domínio a partir do valor dos potenciais elétricos medidos. A rede recebe o valor dos potenciais e calcula o valor
da condutividade no interior do domínio, e assim permitindo criar uma imagem do interior com o auxílio do programa Gmsh.

# Resultados
A seguir a esquerda temos a imagem que queremos, e a direita o que foi calculado pela rede.

<img src="./results/true/true_0_0_001.png" alt="True" width="250" /> <img src="./results/predicted/0_modified_predictMesh.jpeg" alt="Predicted" width="250" />

# Agradecimentos
Financiado com uma bolsa PIBIC de Agosto de 2018 até Março de 2019, e depois financiado pela FAPESP de Março de 2019
até Fevereiro de 2020.
