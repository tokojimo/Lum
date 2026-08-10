# Validation du pipeline cinétique historique

## Statut et limite de portée

L'équivalence avec le pipeline historique **n'est pas encore démontrée**. Les
scripts historiques sont disponibles dans `examples/synthetic/`, mais le dépôt
ne contient ni classeur Varioskan représentatif, ni export de référence produit
par ces scripts. Un test construit uniquement à partir des nouvelles fonctions
validerait leur cohérence interne, pas leur équivalence scientifique.

La prochaine validation doit couvrir, sans court-circuiter d'étape :

```text
import → QC → blancs → normalisation → cinétique
```

Les données réelles doivent être anonymisées avant tout commit. Il faut en
particulier retirer les noms de projets, opérateurs, chemins locaux, commentaires
libres, identifiants d'échantillons et propriétés de document. Les valeurs de
mesure ne doivent être transformées que si cette transformation est documentée
et appliquée de la même manière aux résultats attendus.

## Paquet de référence requis

Le paquet minimal à déposer dans `tests/data/synthetic/` comprend :

1. `varioskan_historical_anonymized.xlsx`, avec ses feuilles de mesure et son
   plan de plaque ;
2. `kinetics_historical_expected.csv`, calculé avec l'ancien pipeline avant
   modification manuelle ;
3. `historical_fixture_manifest.json`, indiquant la version du script, la
   feuille de luminescence retenue, les paramètres, les unités, la méthode
   d'anonymisation et le hash SHA-256 des deux fichiers précédents ;
4. si le pipeline historique exclut ou avertit des séries,
   `kinetics_historical_status.csv`, qui consigne leur statut et leur motif.

Chaque ligne de résultat attendu doit identifier sans ambiguïté
`experience_id`, la condition (`souche` et `Groupe`), `replicat`,
`sample_header` et `puits`. `replicat` reste une identité technique conservée
séparément tant que sa signification biologique n'a pas été validée.

## Colonnes attendues

Le CSV de référence doit exposer au minimum :

| Colonne | Définition |
|---|---|
| `experience_id`, `souche`, `Groupe`, `replicat`, `sample_header`, `puits` | clé de série |
| `od_max`, `od_max_time_h` | maximum de `DO_corr` et premier temps correspondant |
| `od_auc` | AUC trapézoïdale de `DO_corr` |
| `max_growth_rate_per_h` | pente maximale de `log(DO_corr)` en h⁻¹ |
| `max_growth_rate_start_h`, `max_growth_rate_end_h` | fenêtre retenue |
| `growth_rate_r_squared` | R² de cette régression |
| `doubling_time_h` | `log(2) / max_growth_rate_per_h` si la pente est positive |
| `lum_norm_peak`, `lum_norm_peak_time_h` | pic de `Lum_norm` et premier temps correspondant |
| `lum_norm_auc` | AUC trapézoïdale de `Lum_norm` |
| `status`, `reason` | `analyzed`, `rejected` ou `warned`, avec motif explicite |

Les fichiers doivent préserver suffisamment de décimales pour éviter qu'un
arrondi d'export ne masque un écart de calcul.

## Protocole du test de régression

Le futur `tests/test_kinetics_historical.py` devra :

1. contrôler les hashes et les colonnes du paquet de référence ;
2. importer le classeur avec une feuille de luminescence explicitement choisie ;
3. fixer `experience_id` et toutes les décisions de QC et de blancs dans le
   test, sans interaction utilisateur ;
4. exécuter successivement QC, correction des blancs, normalisation et
   `run_kinetics()` ;
5. comparer d'abord l'ensemble exact des clés et des statuts, puis chaque
   métrique numérique ;
6. produire un tableau d'écarts comportant valeur historique, valeur LuxPlate,
   écart absolu, écart relatif, tolérance et verdict ;
7. échouer sur toute série manquante, supplémentaire ou silencieusement
   agrégée, même si les moyennes globales sont proches.

Les paramètres permissifs actuels (`3` points, durée `0 h`, R² `0`, AUC à
partir de `2` points) doivent être enregistrés dans le manifeste, et non
considérés comme des recommandations biologiques. Toute future valeur par
défaut validée scientifiquement devra créer une nouvelle référence explicite.

## Tolérances initiales à faire valider

Une comparaison numérique doit utiliser `numpy.testing.assert_allclose` avec
des tolérances déclarées par métrique. Comme point de départ technique, avant
validation scientifique :

| Famille | `rtol` | `atol` |
|---|---:|---:|
| temps (h) | `0` | `1e-9` |
| DO, taux, R², temps de doublement | `1e-7` | `1e-10` |
| luminescence et AUC | `1e-7` | `1e-8` |

Ces seuils ne doivent pas servir à accepter une différence de convention
(fenêtre, gestion des valeurs manquantes, doublons, règle du pic ou AUC). Un
écart systématique exige une explication et, si nécessaire, des modes
`legacy` et `corrected` distincts plutôt qu'une tolérance élargie.

## Critères de sortie du blocage

L'équivalence historique ne peut être revendiquée que lorsque le paquet est
présent, que le test traverse réellement les cinq étapes, que toutes les clés
et décisions concordent, et que le rapport d'écarts est versionné. Jusqu'alors,
les tests synthétiques existants valident le contrat LuxPlate uniquement.
