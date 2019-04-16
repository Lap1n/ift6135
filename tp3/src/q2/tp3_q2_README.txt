J’ai fait une bonne partie de la question 2. Voici ce que je vais faire dans les prochains jours. 

1) Vérifier le code, surtout le code pour les questions 2.2.1 et 2.2.2, car c’est très facile de faire des erreurs là-dedans (il y en a très possiblement). Il y a beaucoup de calculs avec des petits détails, ce n’est pas si évident. Pour le moment, j’obtiens plus de -95 pour le ELBO sur le validation et le test après 20 epochs, ce qui est supérieur à -96. Pour la question 2.2.1, j’obtiens des valeurs de -88.5 environ pour le log likelihood estimate sur le validation et le test. Je ne sais pas si ça fait du sens par rapport au -94.75 (ELBO). Peut-être qu’il faudrait demander aux tas.

2) Je dois changer un peu le code pour les méthodes “importance_sampling” (2.2.1) et “all_log_X” (2.2.2), car pour le moment la première méthode prend des argument X de la forme (batchsize, 1, 28, 28), mais ils veulent (batchsize, 728), ce qui est un peu embêtant étant donné que l’on utilise un CNN, mais je vais le changer pour répondre aux critères

3) Nettoyer le code.

4) Pour la question 2.2.2, j’ai tout dans le code, mais je ne sais pas s’ils veulent les valeurs du ‘log likelihood estime average’ pour le validation et le test uniquement après 20 epochs, ou s’il veulent ces valeurs après chaque epoch pour faire un graphe. La même question se posent pour les ELBO du validation et du test.

Je suis très ouvert à vos commentaires/changements/suggestions. 
