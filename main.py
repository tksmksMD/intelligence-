import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import estimate_bandwidth

# Завантаження даних з файлу
file_path = 'C:/Users/Саня/PycharmProjects/laba1/lab1.txt'
data = np.loadtxt(file_path, delimiter=';')

# Відображення вихідних точок
plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], edgecolors='black', facecolors='none')
plt.title('Вихідні точки')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')

# Метод зсуву середнього для визначення кількості кластерів
bandwidth = estimate_bandwidth(data, quantile=0.2)
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(data)
num_clusters = len(np.unique(meanshift.labels_))

# Відображення центрів кластерів (метод зсуву середнього)
plt.figure(2)
plt.scatter(data[:, 0], data[:, 1], edgecolors='black', facecolors='none')
plt.title('Вихідні точки з центрами кластерів (метод зсуву середнього)')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')

# Виведемо всі центри
for center in meanshift.cluster_centers_:
    plt.scatter(center[0], center[1], marker='x', s=50, color='red', linewidth=1, label='Центр кластеру')

plt.legend()

# Оцінка score для різної кількості кластерів
scores = []
for i in range(2, 16):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=1)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    scores.append(score)

# Відображення бар діаграмми score(number of clusters)
plt.figure(3)
plt.bar(range(2, 16), scores, color='skyblue', edgecolor='black')
plt.title('Бар діаграмма score(number of clusters)')
plt.xlabel('Кількість кластерів')
plt.ylabel('Score')

# Кластеризація методом k-середніх з оптимальною кількістю кластерів
optimal_clusters = np.argmax(scores) + 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=1)
kmeans.fit(data)

# Визначення границь та відображення кластеризованих даних
h = .02  # Параметр для визначення шагу мешу
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Отримання міток для кожної точки
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Перетворення розмірності для відображення контурів
Z = Z.reshape(xx.shape)

# Відображення контурів та кластеризованих даних з областями кластерізації
plt.figure(4)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, edgecolors='black', facecolors='none', linewidths=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=50, color='red', linewidth=2, label='Центри кластерів')
plt.title('Кластеризовані дані з границями кластерів')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.legend()
plt.show()
