# Application Web de Gestion de Données de Machine Learning

## 📝 Description

Cette application web permet aux utilisateurs de gérer et d'entraîner des modèles de machine learning à travers une interface utilisateur intuitive. Elle intègre des fonctionnalités de versioning des données et des modèles avec DVC, ainsi qu'un pipeline d'intégration continue avec Jenkins.

## 🎯 Fonctionnalités

- 📤 Upload de données (formats CSV, XLSX, XLS)
- 🔄 Préparation et prétraitement des données
- 🤖 Sélection et entraînement de modèles de machine learning
- 📊 Visualisation des résultats d'entraînement
- 🔄 Versioning des données et modèles avec DVC
- 🔄 Intégration continue avec Jenkins
- 📱 Interface utilisateur moderne et responsive

## 🛠️ Technologies Utilisées

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS (Tailwind/DaisyUI)
- **Base de données**: Pandas pour la gestion des données
- **Machine Learning**: Scikit-learn
- **Versioning**: Git, GitHub, DVC
- **CI/CD**: Jenkins
- **Stockage Cloud**: Google Drive (via Google Cloud Console)

## 📋 Prérequis

- Python 3.8+
- Git
- DVC
- Compte Google Cloud avec accès à Google Drive
- Jenkins (pour l'intégration continue)

## 🔧 Installation

1. **Cloner le repository**

```bash
git clone [URL_DU_REPO]
cd ml-web-app
```

2. **Créer un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**
   Créer un fichier `.env` à la racine du projet avec les variables suivantes :

```env
APP_NAME=nom de l'application
SECRET_KEY=votre_clé_secrète
MAX_CONTENT_LENGTH=16777216  # 16MB en bytes
UPLOAD_FOLDER=data/raw       # path vers les donnees a versionner par DVC
MODEL_FOLDER=models          # path vers les models a versionner par DVC
DVC_REMOTE_URL=remote dvc
GOOGLE_APPLICATION_CREDENTIALS=chemin/vers/votre/fichier-credentials.json
```

5. **Configurer DVC**

```bash
dvc init
dvc remote add -d myremote gdrive://[ID_DU_DOSSIER_GDRIVE]
```

## 🚀 Utilisation

1. **Démarrer l'application**

```bash
python -m app.main
```

2. **Accéder à l'application**

- Ouvrir un navigateur et aller à `http://localhost:5000`

3. **Utiliser l'application**
   - Uploader un fichier de données
   - Sélectionner les features et le target
   - Choisir le modèle et les paramètres
   - Lancer l'entraînement
   - Visualiser les résultats
   - Sauvegarder le modèle

## 🧪 Tests

### Prérequis pour les Tests

- Avoir installé toutes les dépendances
- Avoir configuré l'environnement virtuel
- Avoir configuré les variables d'environnement

### Exécution des Tests

1. **Lancer tous les tests**

```bash
pytest
```

2. **Lancer les tests avec couverture**

```bash
pytest --cov=app tests/
```

3. **Lancer les tests avec rapport HTML**

```bash
pytest --cov=app --cov-report=html tests/
```

4. **Lancer les tests en mode verbeux**

```bash
pytest -v
```

### Structure des Tests

Les tests sont organisés dans le dossier `tests/` avec la structure suivante :

```
tests/
├── test_data_handler.py    # Tests pour la gestion des données
└── test_model_handler.py   # Tests pour la gestion des modèles
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Fork le projet
2. Créer une branche pour votre feature (`git checkout -b feature/feature_name`)
3. Commit vos changements (`git commit -m 'Add some feature_name'`)
4. Push vers la branche (`git push origin feature/feature_name`)
5. Ouvrir une Pull Request

## 👥 Contributeurs

### Équipe de Développement

| Nom                               | 
| --------------------------------- | 
| Paul Alognon-Anani                | 
| Castelnau Godefroy Ondongo        | 
| Amadou Tidiane Diallo             | 
| Mayombo Abel M.O                  | 
| Joan-Yves Darys Anguilet          | 

## 📚 Spécifications du Projet

### Objectifs

- Versioning du code source avec Git et GitHub
- Versioning des données et modèles avec DVC
- Mise en place d'un pipeline CI avec Jenkins

### Fonctionnalités Requises

- Upload de données (CSV, XLSX, XLS)
- Sélection et entraînement de modèles ML
- Visualisation des résultats
- Versioning des données et modèles
- Tests automatisés