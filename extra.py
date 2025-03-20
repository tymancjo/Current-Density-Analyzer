
from dataclasses import dataclass

@dataclass
class PRO:
    is_pro: bool = True

@dataclass
class osoba:
    name: str
    is_pro: bool = False



JA: osoba = osoba('Tomasz Tomanek')
PROGRAMISTA: PRO = PRO()


if __name__ == "__main__":
    print(JA != PROGRAMISTA)





