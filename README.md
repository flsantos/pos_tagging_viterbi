
### Código

* Os códigos encontram-se dentro da pasta src.


### Pacotes utilizados

-numpy
-pandas
-pickle
-nltk
-timeit
-sklearn
-logging


* Execução do código

1> python preprocess.py

Esse código é responsável por ler a base inicial "CETENFolha-1.0_jan2014.cg" e gerar um arquivo pickle pré-processado contendo todas as sentenças em um formato de sequências de tuplas (palavra, etiqueta).
Para rodar esse código, é necessário que o arquivo de entrada "CETENFolha-1.0_jan2014.cg" esteja na mesma pasta do código fonte.



2> python evaluate.py

Esse código é responsável por ler o arquivo de setenças pré-processado gerado pelo código anterior "preprocessed_CETEN_v2.pkl" e rodar os modelos desenvolvidos através de uma validaçao cruzada de 5 pastas.
No fim, o resultado da acurácia de cada pasta é salvo no arquivo evaluation.log, no seguinte formato: [corretas_viterbi, corretas_baseline, totais, elapsed]
	-corretas_viterbi: número de etiquetas classificadas corretamente pelo modelo de Viterbi
	-corretas_baseline: número de etiquetas classificadas corretamente pelo modelo baseline
	-totais: número total de etiquetas
	-elapsed: tempo em segundos decorrido para rodar essa pasta



* Notebooks


Seguem também como fontes do projeto os jupyter notebooks utilizados para desenvolvimento e testes do programa.
Não são necessários para execução do código, pois todo ele foi passado para os fontes preprocess.py, evaluate.py e viterbi.py, porém estão disponíveis para caso seja necessária uma maior exploração do código.