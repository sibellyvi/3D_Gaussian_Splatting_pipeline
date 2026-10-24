# Reconstrução 3D da ampola com Gaussian Splatting

## 1. Objetivo

Este documento registra o processo de reconstrução tridimensional de uma ampola usando **3D Gaussian Splatting (3DGS)**, desde os frames extraídos do vídeo até o treinamento e a geração do arquivo final `point_cloud.ply`.

O objetivo é manter um registro reproduzível do fluxo de trabalho, incluindo os comandos utilizados, os problemas encontrados e as soluções aplicadas.

> **Nota:** este documento descreve a execução realizada no ambiente local. Informações que não foram medidas ou confirmadas — como duração exata do vídeo, dispositivo de captura e métricas finais de qualidade — são identificadas como não disponíveis, em vez de serem estimadas.

## 2. Ferramentas e ambiente

| Componente | Configuração |
|---|---|
| Sistema de comandos | Windows PowerShell |
| Projeto 3DGS | `C:\Projetos3D\gaussian-splatting` |
| Ambiente Python | `C:\Users\eronp\.conda\envs\gaussian_splatting\python.exe` |
| Executável COLMAP | `C:\Projetos3D\colmap\bin\colmap.exe` |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB de VRAM |
| Memória RAM instalada | Aproximadamente 32 GB |
| Entrada de imagens | `C:\Projetos3D\ampola_gs\input` |
| Backup do modelo COLMAP | `C:\Projetos3D\ampola_gs\modelo_398` |
| Modelo corrigido para treinamento | `C:\Projetos3D\ampola_gs_corrigido` |
| Pasta de saída do treinamento | `C:\Projetos3D\gaussian-splatting\output\ampola` |

## 3. Dados de entrada

Foram utilizados **398 frames JPEG** extraídos do vídeo da ampola.

- Pasta dos frames originais: `C:\Projetos3D\ampola_gs\input`
- Quantidade de imagens encontrada: **398**
- Dimensões observadas em uma imagem da pasta corrigida: **4299 × 7647 pixels** (orientação vertical).

A dimensão registrada corresponde à imagem inspecionada na pasta corrigida. A resolução original do vídeo e o dispositivo utilizado para a captura não foram registrados nesta documentação.

### Por que a resolução das imagens foi importante?

As imagens eram muito grandes para serem carregadas simultaneamente na GPU com a configuração inicial. Isso provocou erros de falta de memória CUDA durante o carregamento das câmeras.

Não foi necessário extrair novamente os frames. O ajuste foi feito na etapa de treinamento, reduzindo a resolução de processamento.

## 4. Reconstrução com COLMAP

O COLMAP foi utilizado para estimar as câmeras e produzir a reconstrução esparsa necessária para inicializar o treinamento 3DGS.

### 4.1. Estrutura de modelos encontrada

Na reconstrução original, foram encontrados dois modelos:

- `distorted\sparse\0`: continha apenas **2 imagens**.
- `distorted\sparse\1`: continha o modelo com **398 imagens registradas**.

O problema foi que o script `convert.py` selecionava automaticamente `distorted/sparse/0`. Assim, o script apontava para o modelo de apenas duas imagens, em vez do modelo completo.

A causa exata que levou o COLMAP a produzir esses dois modelos não foi determinada. O que foi confirmado é que o modelo completo estava no diretório `sparse/1`, e não no `sparse/0`.

### 4.2. Backup do modelo

O modelo COLMAP com as 398 imagens foi mantido em:

```text
C:\Projetos3D\ampola_gs\modelo_398
```

Manter esse backup permitiu executar novamente a etapa de undistortion sem refazer toda a reconstrução.

### 4.3. Undistortion do modelo completo

Foi executado o comando abaixo no PowerShell:

```powershell
& "C:\Projetos3D\colmap\bin\colmap.exe" image_undistorter `
  --image_path "C:\Projetos3D\ampola_gs\input" `
  --input_path "C:\Projetos3D\ampola_gs\modelo_398" `
  --output_path "C:\Projetos3D\ampola_gs_corrigido" `
  --output_type COLMAP
```

Essa etapa gerou as imagens corrigidas e os arquivos do modelo COLMAP no diretório de saída.

### 4.4. Conversão do modelo para TXT

Para inspecionar o modelo em formato legível, também foi executada a conversão para TXT:

```powershell
& "C:\Projetos3D\colmap\bin\colmap.exe" model_converter `
  --input_path "C:\Projetos3D\ampola_gs_corrigido\sparse" `
  --output_path "C:\Projetos3D\ampola_gs_corrigido\sparse" `
  --output_type TXT
```

Os arquivos `.bin` são versões binárias do modelo COLMAP; os `.txt` contêm representações textuais dos dados.

### 4.5. Organização da pasta `sparse/0`

O resultado da etapa anterior deixou os arquivos do modelo diretamente em `sparse`, sem a subpasta `sparse\0` esperada pelo fluxo de treinamento utilizado. Para adequar a estrutura, foi criada a pasta `sparse\0` e copiados para ela os arquivos `.bin` e `.txt`:

```text
C:\Projetos3D\ampola_gs_corrigido\sparse\0\
```

Foram confirmados estes dez arquivos:

```text
cameras.bin
cameras.txt
frames.bin
frames.txt
images.bin
images.txt
points3D.bin
points3D.txt
rigs.bin
rigs.txt
```

A estrutura final passou a conter os arquivos esperados em `sparse\0`.

### 4.6. Validação da reconstrução

O modelo foi analisado com:

```powershell
& "C:\Projetos3D\colmap\bin\colmap.exe" model_analyzer `
  --path "C:\Projetos3D\ampola_gs_corrigido\sparse"
```

Resultados registrados:

| Métrica COLMAP | Valor |
|---|---:|
| Rigs | 1 |
| Câmeras | 1 |
| Frames | 398 |
| Frames registrados | 398 |
| Imagens | 398 |
| Imagens registradas | 398 |
| Pontos 3D esparsos | 40.558 |
| Observações | 222.819 |
| Comprimento médio das trilhas | 5,493836 |
| Observações médias por imagem | 559,846734 |
| Erro médio de reprojeção | 1,371849 px |

Todas as **398 imagens foram registradas** no modelo validado. O COLMAP produziu **40.558 pontos 3D esparsos**, que foram utilizados como base de inicialização para o treinamento.

O erro médio de reprojeção de 1,371849 px é uma medida geométrica do ajuste entre observações nas imagens e a reconstrução estimada. Ele não mede, sozinho, a qualidade visual final do modelo Gaussian Splatting.

## 5. Treinamento com 3D Gaussian Splatting

### 5.1. Problemas de memória encontrados

Foram feitas tentativas de treinamento com diferentes configurações.

**Tentativa 1 — resolução padrão (`-r 1`)**

```powershell
& "C:\Users\eronp\.conda\envs\gaussian_splatting\python.exe" train.py `
  -s "C:\Projetos3D\ampola_gs_corrigido" `
  -m "C:\Projetos3D\gaussian-splatting\output\ampola" `
  -r 1 --antialiasing --eval --disable_viewer
```

O processo carregou as 398 câmeras, mas ocorreu um erro de falta de memória CUDA durante o carregamento das imagens.

**Tentativa 2 — `-r 2`**

```powershell
& "C:\Users\eronp\.conda\envs\gaussian_splatting\python.exe" train.py `
  -s "C:\Projetos3D\ampola_gs_corrigido" `
  -m "C:\Projetos3D\gaussian-splatting\output\ampola" `
  -r 2 --antialiasing --eval --disable_viewer
```

Essa tentativa também apresentou falta de memória CUDA.

**Por que `-r 2` não resolveu?**

Na implementação utilizada, os valores `1`, `2`, `4` e `8` funcionam como fatores de redução. Assim, `-r 2` reduz as dimensões aproximadamente pela metade. Para imagens de 4299 × 7647 pixels, isso ainda deixa imagens muito grandes, de aproximadamente 2149 × 3823 pixels.

**Tentativa 3 — armazenamento das imagens na CPU**

```powershell
& "C:\Users\eronp\.conda\envs\gaussian_splatting\python.exe" train.py `
  -s "C:\Projetos3D\ampola_gs_corrigido" `
  -m "C:\Projetos3D\gaussian-splatting\output\ampola" `
  -r 2 --data_device cpu --antialiasing --eval --disable_viewer
```

Essa configuração avançou além do erro de memória CUDA, mas terminou com `MemoryError` durante o redimensionamento/carregamento das imagens na memória do sistema.

### 5.2. Configuração que funcionou

A solução foi definir uma largura de processamento de 1600 pixels e manter os dados das imagens na CPU:

```powershell
& "C:\Users\eronp\.conda\envs\gaussian_splatting\python.exe" train.py `
  -s "C:\Projetos3D\ampola_gs_corrigido" `
  -m "C:\Projetos3D\gaussian-splatting\output\ampola" `
  -r 1600 --data_device cpu --antialiasing --eval --disable_viewer
```

O treinamento foi concluído com sucesso, gerando o modelo da iteração 30.000.

### 5.3. O que significam os parâmetros?

| Parâmetro | Função |
|---|---|
| `-s` | Define a pasta de origem, com imagens e dados COLMAP. |
| `-m` | Define a pasta em que os resultados do treinamento serão salvos. |
| `-r 1600` | Define uma largura de processamento de 1600 pixels nesta implementação, preservando a proporção da imagem. |
| `--data_device cpu` | Mantém os dados das imagens na memória do sistema, reduzindo a pressão sobre a VRAM durante o carregamento. O treinamento continua utilizando a GPU para os cálculos. |
| `--antialiasing` | Ativa a opção de antialiasing disponível no código de treinamento utilizado. |
| `--eval` | Ativa a avaliação com separação de imagens para treino e teste. |
| `--disable_viewer` | Desativa o visualizador durante o treinamento. |
| `sh_degree=3` | Grau dos coeficientes de aparência usados no modelo, conforme a configuração registrada. |

**Atenção à opção `-r`:** neste código, `-r 1600` é interpretado como uma resolução-alvo, e não como um fator de redução por 1600. Por isso, ele resolveu o problema que persistia com `-r 2`.

### 5.4. Configuração registrada pelo treinamento

O arquivo `cfg_args` do resultado contém:

```text
Namespace(
  sh_degree=3,
  source_path='C:\\Projetos3D\\ampola_gs_corrigido',
  model_path='C:\\Projetos3D\\gaussian-splatting\\output\\ampola',
  images='images',
  depths='',
  resolution=1600,
  white_background=False,
  train_test_exp=False,
  data_device='cpu',
  eval=True
)
```

Esse arquivo registra os principais argumentos associados ao modelo treinado.

## 6. Resultados

### 6.1. Modelo final

O modelo final foi salvo em:

```text
C:\Projetos3D\gaussian-splatting\output\ampola\point_cloud\iteration_30000\point_cloud.ply
```

Informações confirmadas:

| Item | Resultado |
|---|---:|
| Iteração final | 30.000 |
| Quantidade de Gaussianos (`element vertex`) | 473.877 |
| Tamanho do PLY final | 117.523.027 bytes |
| Tamanho aproximado em MB decimais | 117,52 MB |
| Tamanho aproximado em MiB | 112,03 MiB |

O cabeçalho do arquivo PLY contém:

```text
format binary_little_endian 1.0
element vertex 473877
```

O número de vértices representa a quantidade de primitivas Gaussianas armazenadas no arquivo final. Não deve ser confundido com os 40.558 pontos esparsos do COLMAP.

### 6.2. Checkpoint intermediário

Também foi encontrado o arquivo:

```text
C:\Projetos3D\gaussian-splatting\output\ampola\point_cloud\iteration_7000\point_cloud.ply
```

| Item | Resultado |
|---|---:|
| Iteração | 7.000 |
| Tamanho | 100.227.507 bytes |
| Tamanho aproximado | 100,23 MB |

Esse arquivo é um resultado intermediário, não o modelo final.

### 6.3. Métricas de avaliação disponíveis

No log da iteração 7.000 foram registradas as seguintes métricas:

| Conjunto | L1 | PSNR |
|---|---:|---:|
| Teste | 0,02171523 | 28,0557 dB |
| Treino | 0,01861567 | 29,2348 dB |

Esses valores correspondem à **iteração 7.000**, não à iteração final 30.000. Portanto, devem ser apresentados apenas como resultados intermediários.

Não há, neste registro, valores finais confirmados de PSNR, SSIM ou LPIPS para a iteração 30.000. Eles não devem ser inferidos a partir do tamanho do arquivo ou da quantidade de Gaussianos.

### 6.4. Tempo de treinamento

O log registra o início da execução às 15:29:21 e um progresso de 7.000 iterações com aproximadamente 29 minutos e 24 segundos de tempo decorrido naquele ponto. Como não foi registrado de forma confiável o tempo total da execução até a conclusão, a duração total do treinamento permanece **não determinada**.

## 7. Problemas encontrados e soluções

| Problema | Causa ou evidência observada | Solução aplicada |
|---|---|---|
| `convert.py` selecionava um modelo com apenas duas imagens | O script apontava para `distorted/sparse/0`, que continha 2 imagens; o modelo completo estava em `sparse/1`. | Usar o modelo completo de 398 imagens como entrada para o `image_undistorter`. |
| Estrutura incompatível com o caminho esperado pelo treinamento | Os arquivos do modelo corrigido ficaram diretamente em `sparse`, sem a subpasta `sparse/0`. | Criar `sparse/0` e copiar os arquivos `.bin` e `.txt` do modelo para essa pasta. |
| Falta de memória CUDA durante o carregamento | As imagens de entrada eram muito grandes. | Usar `--data_device cpu` e ajustar a resolução de processamento. |
| `-r 2` não foi suficiente | O parâmetro reduzia as dimensões aproximadamente pela metade, mantendo imagens de alta resolução. | Usar `-r 1600`, que define a largura de processamento desejada. |
| `MemoryError` com as imagens na CPU e `-r 2` | O carregamento/redimensionamento ainda consumia muita memória do sistema. | Reduzir a resolução para 1600 pixels de largura; o treinamento com essa configuração foi concluído. |

## 8. Limitações e próximos passos

Para uma análise mais completa e uma possível publicação científica, recomenda-se complementar este registro com:

1. **Inspeção visual do modelo final:** verificar se a ampola apresenta forma, transparência, contornos e detalhes preservados, registrando imagens da visualização.
2. **Métricas finais:** executar a avaliação do modelo final e registrar PSNR e L1; calcular SSIM e LPIPS somente se houver um procedimento e uma implementação adequados.
3. **Registro da captura:** acrescentar a duração e a resolução do vídeo original, o dispositivo utilizado, as condições de iluminação e como a câmera percorreu o objeto, caso esses dados estejam disponíveis.
4. **Reprodutibilidade:** manter os comandos, as versões das ferramentas e os arquivos de configuração usados.
5. **Cautela na interpretação:** PSNR, L1 e outras métricas quantitativas devem ser analisadas junto com a inspeção visual. O erro de reprojeção do COLMAP mede o ajuste geométrico e não substitui a avaliação visual do modelo 3DGS.

## 9. Resumo do fluxo executado

```text
Vídeo da ampola
      ↓
Extração de 398 frames JPEG
      ↓
Reconstrução de câmeras e pontos esparsos com COLMAP
      ↓
Seleção do modelo COLMAP com 398 imagens
      ↓
Undistortion do modelo completo
      ↓
Organização dos arquivos em sparse/0
      ↓
Validação com model_analyzer
      ↓
Treinamento 3DGS com -r 1600 e --data_device cpu
      ↓
Modelo final: iteration_30000/point_cloud.ply
      ↓
473.877 Gaussianos — arquivo de aproximadamente 117,52 MB
```

## 10. Conclusão

A reconstrução da ampola foi treinada com sucesso usando 398 frames e um modelo COLMAP validado com 398 imagens registradas e 40.558 pontos 3D esparsos. Após corrigir a seleção do modelo COLMAP, organizar a estrutura `sparse/0` e reduzir a resolução de processamento para 1600 pixels, o treinamento atingiu 30.000 iterações.

O arquivo final contém 473.877 Gaussianos e ocupa aproximadamente 117,52 MB. Os resultados quantitativos de treino e teste registrados neste documento são intermediários, da iteração 7.000; a avaliação quantitativa final e a inspeção visual documentada ainda precisam ser acrescentadas.

---

## Apêndice A — Comando final de treinamento

```powershell
& "C:\Users\eronp\.conda\envs\gaussian_splatting\python.exe" train.py `
  -s "C:\Projetos3D\ampola_gs_corrigido" `
  -m "C:\Projetos3D\gaussian-splatting\output\ampola" `
  -r 1600 --data_device cpu --antialiasing --eval --disable_viewer
```

## Apêndice B — Localização dos arquivos principais

| Arquivo ou pasta | Finalidade |
|---|---|
| `C:\Projetos3D\ampola_gs\input` | Frames JPEG de entrada |
| `C:\Projetos3D\ampola_gs\modelo_398` | Backup do modelo COLMAP completo |
| `C:\Projetos3D\ampola_gs_corrigido` | Imagens corrigidas e dados COLMAP preparados para o treinamento |
| `C:\Projetos3D\gaussian-splatting\output\ampola\cfg_args` | Configuração registrada do treinamento |
| `C:\Projetos3D\gaussian-splatting\output\ampola\point_cloud\iteration_7000\point_cloud.ply` | Modelo intermediário |
| `C:\Projetos3D\gaussian-splatting\output\ampola\point_cloud\iteration_30000\point_cloud.ply` | Modelo final |
