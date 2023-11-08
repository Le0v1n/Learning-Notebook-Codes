from rich.console import Console


console = Console()
console.print("Hello", "Le0v1n")  # 一定要定义 style，不然和普通的 print 没区别
console.print("Hello", "Le0v1n", style="bold red")  # 一定要定义 style，不然和普通的 print 没区别

console.print("Where there is a [bold cyan]Will[/bold cyan] there [u]is[/u] a [i]way[/i].")