Dorobienie 10 kerneli, czyli "10" małych, lub mniej dużych, 100 lini kodu - 10 małych róznych funkcji + shared memory, z opencv na obrazach typu sobel. 

4 fajne kernele to tez ok, z shared memory

dorobienie shared memory do kernel_v2 to liczy się jako 3 kernele


-------------------------------------------------------------------------------
Lub 2 opcja


Znalezienie operacji/biliboteki która działa na cpu, albo napisać prostą konwolucję i liniową




CUDA, od zera rozpoznawanie cyferek (ileś razy powielona konwolucja, konwolucja RELU, liniowy, 3 kernele)




conwolucja - batch normalizacja



RESNET5


MNIST


nvidia-compute, occupancy kerneli, profilowanie


fusowanie kerneli, konwolucja - batch - normalizacja - relu








Listę zadań:

Jakiś najprostszt resnet


bez zdroput


conv2




1. Znaleść najrpostszy model do roznawania cyfr (MNIST)
2. Torch, na CPU profilowanie, i czasówki i accuracy i porównać, zrobić to samo na CUDA, GPU
3. Kernele: konwolucja, batch normalizacja, relu, fully connected, każdy poprzedni z każdym poprzednim, (flatern i maks pulling krótkie kernele to też git)
(jakąś śmieszna próbe optymalizacji, fusować więcej kerneli w jeden, turbo speed), jeden duży kernel, i optymalizacje +_ wsysłac we wtorek do oceny co tam dalej można
