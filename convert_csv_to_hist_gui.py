import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("histogram_green.csv")
plt.figure(figsize=(10, 6))
plt.bar(df["Bin"], df["Count"], width=1.0, color='blue')
plt.xlabel("Bin")
plt.ylabel("Count")
plt.title("Histogram")
plt.savefig("histogram_image.jpg")
plt.show()
