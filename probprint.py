from termcolor import colored, cprint

def probprint(text, probability, maxvalue):
    if probability < 0.1*maxvalue:
        return colored(text,"white")
    if probability < 0.2*maxvalue:
        return colored(text,"cyan")
    if probability < 0.4*maxvalue:
        return colored(text,"green")
    if probability < 0.6*maxvalue:
        return colored(text,"yellow")
    if probability < 0.8*maxvalue:
        return colored(text,"red")
    else:
        return colored(text,"grey")
