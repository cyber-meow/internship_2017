## Abstract

## Introduction

* Contexte du stage (environnement, matériel, logiciel...)
* Contexte scientifique (Robotique développementalle, construction d'une
  représentation interne, l'importance de multimodalité)
* Décrire brièvement les travails que j'ai réalisé dans mon stage
  et la structure du rapport

## Related work

* Multimodalité en général
* Apprentissage multimodal dans le cadre de robotique
* Reconnaissance de gestes
* Reconnaissance de la parole

## Presentation of basic network architecture

* CNN
* Auto-encodeur

## Datasets and preprocessing

## Experimental setup

* learning rate
* optimizer
* relu
* batch normalization
* weight regularization

## Single modality

### Classification

* Découpage de dataset
* Parler juste de perceptron et CNN ici
* Détailler les architectures de CNN utilisés
* Overfitting
* Discussion de la performance sur différents datasets

### Convolutional auto-encoder

* Présenter les architectures (nombre de couches, neurones)
* L'apprentissage de la representatoin évalué par la performance
  de classification
* Quelques images pour montrer la reconstruction
* On peut même montrer des images d'embedding de Tensorboard

## Multimodal

### Fusion

* Objectif et hypothèse (présence de plusieurs modalités lors de
  l'apprentissage non-superveisé ou encore de l'apprentissage supervisé
  directement sur plusieurs modalités, une meilleure performance
  pour classification ou d'établir un lien à deux sens entre deux modalités)
* Présenter les structures (Bimodal AE et Bimodal CNN).
* Comparaison, résultats, discussion pour la partie classification
* Récupèrer une modalité à partir de l'autre, reconstruction

### Transfert

* Présenter le problématique
* Présenter la méthode utilisé
* Détailler les expériences (sépration de données etc.)
* Résultat

## Conclustion
