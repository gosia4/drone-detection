import os
import cv2

def zmien_nazwy_plikow0(folder):
  """
  Funkcja zmienia nazwy plików w podanym folderze na format "test1_frame_x.txt",
  gdzie x to kolejne numery plików.

  Argumenty:
    folder (str): Ścieżka do folderu z plikami.
  """

  pliki = os.listdir(folder)

  for i, plik in enumerate(pliki):
    nowa_nazwa = f"free_2_frame_{i}.jpg"
    os.rename(os.path.join(folder, plik), os.path.join(folder, nowa_nazwa))


# Przykład użycia
folder = (r"C:\Users\gosia\OneDrive - vus.hr\Desktop\drone-detection-thermal-images\thermal_signature_drone_detection\thermographic_data\test\images\free_2")
zmien_nazwy_plikow0(folder)

#jako drugą funkcję użyj tej
def zmien_nazwy_plikow(folder):
  """
  Funkcja zmienia nazwy plików w podanym folderze na format "free_1_frame_x.jpg",
  gdzie x to kolejne numery plików z odpowiednim zero-paddingiem.

  Argumenty:
      folder (str): Ścieżka do folderu z plikami.
  """
  pliki = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

  pliki.sort()

  # Znajdź maksymalną długość numeru, aby ustalić zero-padding
  max_length = len(str(len(pliki)))

  for i, plik in enumerate(pliki):
    nowa_nazwa = f"free_3_frame_{str(i).zfill(max_length)}.jpg"
    os.rename(os.path.join(folder, plik), os.path.join(folder, nowa_nazwa))


# Przykład użycia
# folder = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\test\images\free_2"
# zmien_nazwy_plikow(folder)

import re


#tej funkcji najpierw użyć, żeby sortowanie się nie popsuło
def zmien_nazwy_plikow3(folder):
  """
  Funkcja zmienia nazwy plików w podanym folderze, dodając wiodące zera do numerów plików mniejszych niż 100.

  Argumenty:
      folder (str): Ścieżka do folderu z plikami.
  """
  pliki = os.listdir(folder)

  for plik in pliki:
    # Dopasowanie numeru w nazwie pliku
    match = re.search(r'(\d+)', plik)
    if match:
      numer = int(match.group(1))
      # Dodaj wiodące zera, jeśli numer jest mniejszy niż 100
      nowy_numer = f'{numer:03}' if numer < 100 else str(numer)
      nowa_nazwa = plik.replace(match.group(1), nowy_numer)
      os.rename(os.path.join(folder, plik), os.path.join(folder, nowa_nazwa))


# Przykład użycia
# folder = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\validate\images\free_3"
# zmien_nazwy_plikow3(folder)


def add_zero_to_files(directory):
  # Przechodzimy przez wszystkie pliki w podanym katalogu
  for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Tylko pliki .txt
      filepath = os.path.join(directory, filename)
      with open(filepath, 'r') as file:
        lines = file.readlines()  # Czytamy wszystkie linie z pliku

      with open(filepath, 'w') as file:
        for line in lines:
          line = line.strip()  # Usuwamy białe znaki z końców linii
          modified_line = f"0 {line}\n"  # Dodajemy zero na początku linii
          file.write(modified_line)  # Zapisujemy zmodyfikowaną linię do pliku


# Użycie funkcji
# directory_path = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\train\labels\free_1"
# add_zero_to_files(directory_path)


#zmiana z 3-cyfrowych na jedno i dwucyfrowe:

# Ścieżka do folderu zawierającego pliki
# folder_path = "ścieżka/do/twojego/folderu"
#
# # Iteracja przez pliki w folderze
# for filename in os.listdir(folder_path):
#     # Sprawdzenie, czy nazwa pliku zawiera trzycyfrowy numer
#     if filename.isdigit() and len(filename) == 3:
#         # Pobranie liczby z nazwy pliku
#         number = int(filename)
#         # Tworzenie nowej nazwy pliku z jedno- lub dwucyfrową liczbą
#         new_filename = f"{number:02d}" if number < 100 else f"{number:01d}"
#         # Pełna ścieżka do starej nazwy pliku
#         old_path = os.path.join(folder_path, filename)
#         # Pełna ścieżka do nowej nazwy pliku
#         new_path = os.path.join(folder_path, new_filename)
#         # Zmiana nazwy pliku
#         os.rename(old_path, new_path)

# Ścieżka do folderu zawierającego pliki
folder_path = "C:/Users/gosia/drone-detection/drone-detection/thermographic_data/validate/images/free_3"

# Iteracja przez pliki w folderze
# for filename in os.listdir(folder_path):
#   if filename.startswith("free_3_frame_") and filename.endswith(".jpg"):
#     # Pobranie liczby z nazwy pliku
#     number_part = filename[len("free_3_frame_"):-len(".jpg")]
#
#     # Sprawdzenie, czy część liczbową można przekonwertować na liczbę całkowitą
#     if number_part.isdigit():
#       number = int(number_part)
#       # Tworzenie nowej nazwy pliku z jedno- lub dwucyfrową liczbą
#       new_filename = f"free_3_frame_{number}.jpg"
#       # Pełna ścieżka do starej nazwy pliku
#       old_path = os.path.join(folder_path, filename)
#       # Pełna ścieżka do nowej nazwy pliku
#       new_path = os.path.join(folder_path, new_filename)
#       # Zmiana nazwy pliku
#       os.rename(old_path, new_path)
#       print(f"Renamed {filename} to {new_filename}")

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
