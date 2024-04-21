import os
import cv2

def zmien_nazwy_plikow(folder):
  """
  Funkcja zmienia nazwy plików w podanym folderze na format "test1_frame_x.txt",
  gdzie x to kolejne numery plików.

  Argumenty:
    folder (str): Ścieżka do folderu z plikami.
  """

  pliki = os.listdir(folder)

  for i, plik in enumerate(pliki):
    nowa_nazwa = f"free3_frame_{i+48}.txt"
    os.rename(os.path.join(folder, plik), os.path.join(folder, nowa_nazwa))


# Przykład użycia
folder = (r"C:\Users\gosia\OneDrive - vus.hr\Desktop\drone-detection\drone-detection\thermographic_data\validate\labels\przekonwertowane")
zmien_nazwy_plikow(folder)


# Ścieżka do nagrania wideo
# video_path = r"C:\Users\gosia\OneDrive - vus.hr\dodatkowe_dane_z_przyciete_35.mp4"
# output_folder = r"C:\Users\gosia\OneDrive - vus.hr\dodatkowe_dane_z_przyciete_35"  # Ścieżka do folderu, gdzie zostaną zapisane obrazy

# # Utworzenie folderu, jeśli nie istnieje
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Utworzenie obiektu VideoCapture
# cap = cv2.VideoCapture(video_path)
#
# # Licznik klatek
# frame_count = 0
#
# # Odczytywanie kolejnych klatek z nagrania i zapisywanie ich jako obrazy
# while True:
#     ret, frame = cap.read()  # Odczytanie kolejnej klatki
#     if not ret:
#         break  # Jeśli nie ma więcej klatek, zakończ pętlę
#
#     # Zapisanie klatki jako obrazu w folderze wyjściowym
#     frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
#     cv2.imwrite(frame_filename, frame)
#
#     frame_count += 1
#
# # Zwolnienie zasobów
# cap.release()
#
# # Wyświetlenie liczby zapisanych obrazów
# print("Liczba zapisanych obrazów:", frame_count)
