import os

def zmien_nazwy_plikow(folder):
  """
  Funkcja zmienia nazwy plików w podanym folderze na format "test1_frame_x.txt",
  gdzie x to kolejne numery plików.

  Argumenty:
    folder (str): Ścieżka do folderu z plikami.
  """

  pliki = os.listdir(folder)

  for i, plik in enumerate(pliki):
    nowa_nazwa = f"free3_frame_{i}.jpg"
    os.rename(os.path.join(folder, plik), os.path.join(folder, nowa_nazwa))


# Przykład użycia
folder = (r"C:\Users\gosia\OneDrive - vus.hr\Desktop\drone_detection\thermographic_data\validate\labels\free_3")
zmien_nazwy_plikow(folder)
