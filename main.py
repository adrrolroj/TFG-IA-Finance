import re
import inquirer

from finance_utils import update_all_data
from graphic_utils import show_graphic_from_finance


print('¡Bienvenido!')
options = [
    inquirer.List(
        "Option",
        message="¿Que desea hacer?",
        choices=["Actualizar la base de datos de todos los mercados", "Ver graficas de mercado a tiempo real"],
    ),
]

answer = inquirer.prompt(options)
if answer["Option"] == "Ver graficas de mercado a tiempo real":
    with open('markets.txt') as f:
        lines = f.readlines()
    book = {}
    for line in lines:
        book[line.split("==")[1].rstrip("\n")] = line.split("==")[0]
    markets = [
        inquirer.List(
            "market",
            message="¿Que mercado desea ver?",
            choices=book.keys(),
        ),
        inquirer.Text('number', message="¿Cuantas velas deseeas visualizar?", validate=lambda _, x: re.match('^[1-9][0-9]*$', x)),
    ]
    answer = inquirer.prompt(markets)
    file = "data/" + book.get(answer["market"]) + "-5m.csv"
    show_graphic_from_finance(file, int(answer["number"]), book.get(answer["market"]))
else:
    update_all_data()
    print("Todos los datos han sido actualizados con exito")
