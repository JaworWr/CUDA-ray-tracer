# Projekt - ray tracer
## Wymagane biblioteki
* GLFW, GLEW, libGL
* OpenGL Mathematics (GLM)
* yaml-cpp

## Instrukcja użycia
Program uruchamia się poleceniem `<nazwa> <plik z opisem sceny> [początkowa szerokość okna] [początkowa wysokość okna]`, gdzie jako nazwę programu należy podać `ray-tracer-cpu` lub 
`ray-tracer-cuda`. Sterowanie jest następujące:
* Mysz - obrót kamerą
* W,S,A,D - poruszanie się w poziomie
* Q - ruch w górę
* Z - ruch w dół
* M - odblokowanie wskaźnika myszy
* Esc - wyjście z programu

### Format opisu sceny
Sceny podawane są w postaci plików w formacie YAML. Plik powinien zawierać następujące pola:
* `width`, `height` - rozdzielczość sceny (niezależna od rozmiaru okna)
* `fov` - kąt widzenia w pionie
* `bg_color` - opcjonalny kolor tła, domyślnie czarny
* `objects`, `light_sources` - sekwencje opisujące odpowiednio obiekty i źródła światła

Kolory oraz punkty podawane są w postaci list 3-elementowych.

Opis obiektu powinien zawierać następujące pola:
* `type`: rodzaj obiektu, dostępne opcje: `sphere`, `plane`, `dingDong`, `cayley`, `clebsch`, `polynomial`
* `color`: kolor obiektu
* Dodatkowe informacje o obiekcie:
    * Dla sfery: `center`, `radius`
    * Dla płaszczyzny: `origin`, `normal`, opisujące punkt na płaszczyźnie oraz wektor prostpoadły
    * Dla powierzchni opisanych wielomianami: współczynniki podane są jako obiekt `coefficients` zawierający pola postaci `x3`, `y2z`, `xz` itd. opisujące współczynniki przy odpowiednich wyrazach 
      (x^3, y^2z, xz). Ostatni składnik wielomianu podany jest jako współczynnik `c`. Pominiętym współczynnikom nadawana jest wartość 0. 