import numpy as np
from scipy.spatial.distance import mahalanobis

def compute_mahalanobis(polygon1, polygon2):
    poly1 = np.array(polygon1)
    poly2 = np.array(polygon2)

    centroid1 = np.mean(poly1, axis=0)
    centroid2 = np.mean(poly2, axis=0)

    combined = np.vstack((poly1, poly2))
    cov_matrix = np.cov(combined.T)
    inv_cov = np.linalg.inv(cov_matrix)

    distance = mahalanobis(centroid1, centroid2, inv_cov)
    print(f"Mahalanobis Distance: {distance}")

def get_polygon(num):
    print(f"Enter coordinates for Polygon {num} (e.g., x1 y1, x2 y2):")
    points = input("Coordinates: ").strip().split(",")
    return [tuple(map(float, p.strip().split())) for p in points]

polygon1 = get_polygon(1)
polygon2 = get_polygon(2)
compute_mahalanobis(polygon1, polygon2)