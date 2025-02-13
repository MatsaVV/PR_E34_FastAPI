# 📝 Reconnaissance de Chiffres Manuscrits avec IA

Ce projet est une **application de reconnaissance de chiffres manuscrits** utilisant **FastAPI** pour l'API et **Streamlit** pour l'interface utilisateur (repo : Pr_E34_Steamlit).  
L'IA repose sur un modèle entraîné en **Deep Learning** et utilise **MySQL** pour stocker les feedbacks des utilisateurs.

**Déploiement automatique** via **GitHub Actions & Docker** sur **Azure WebApp**.

---

## 📌 Fonctionnalités

✅ API REST pour la reconnaissance de chiffres manuscrits.  
✅ Interface web interactive avec **Streamlit**.  
✅ Base de données **MySQL** pour stocker les feedbacks.  
✅ Sécurisation avec un **token d’authentification**.  
✅ CI/CD avec **GitHub Actions & Docker**.  

---

## 🚀 Déploiement et Automatisation

Le projet est déployé automatiquement grâce à **GitHub Actions et Docker**.  
À chaque **git push origin main**, une **image Docker** est générée et déployée sur **Azure WebApp**.

---

## 🔐 Sécurité

L'API est protégée par un **token d’authentification**.  
Les requêtes doivent inclure un header `x-token` valide pour accéder aux endpoints.

---

## 🛠️ Technologies Utilisées

- **Backend** : FastAPI, TensorFlow/Keras  
- **Frontend** : Streamlit  
- **Base de données** : MySQL  
- **Déploiement** : GitHub Actions, Docker, Azure WebApp  

---

## 📜 Licence

Ce projet est sous licence **MIT**.

