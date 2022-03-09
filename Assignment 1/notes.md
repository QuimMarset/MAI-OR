Podem fer data augmentation per cada imatge un 50% de probabilitat
Cada epoch per cada imatge apliquem un data augmentation
Separar validation set (e.g. 20%) i d'aquestes agafar les de segmentació

Per balancejar el dataset podem preprocessar la segmentació i ordenar-ho per aconseguir un balanceig

Per rotar podem generar una matriu de rotació, centrar la imatge, i multiplicar per la matriu

Per escalar podem retallar la imatge i utilitzar open-cv

-15 +15 graus per exemple
Per escalar entre 0.8 i 1.2