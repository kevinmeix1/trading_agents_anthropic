# VS Code Setup

Follow these steps slowly.

## 1. Open The Folder

Because the `code` terminal command is not installed on your machine, use the app:

1. Open Visual Studio Code.
2. Click `File`.
3. Click `Open Folder...`.
4. Select:

```text
/Users/kaiwenmei/Desktop/quanthack
```

5. Click `Open`.

Optional later: to install the `code` shortcut, open VS Code, press
`Cmd + Shift + P`, type `Shell Command: Install 'code' command in PATH`, and run it.

## 1.5 Install Recommended Extensions

VS Code may show a prompt saying this workspace recommends extensions. Install:

- Python
- Pylance

If no prompt appears:

1. Click the Extensions icon on the left sidebar.
2. Search `Python`.
3. Install the Microsoft Python extension.
4. Search `Pylance`.
5. Install the Microsoft Pylance extension.

## 2. Open The Terminal In VS Code

In VS Code:

1. Click `Terminal`.
2. Click `New Terminal`.

The terminal should open at the project folder. Check with:

```bash
pwd
```

You want:

```text
/Users/kaiwenmei/Desktop/quanthack
```

## 3. Create The Virtual Environment

Run:

```bash
python3.11 -m venv .venv
```

Then activate it:

```bash
source .venv/bin/activate
```

Your terminal prompt should now include `(.venv)`.

If your prompt only shows `(base)`, you are using Anaconda's base Python. This
project needs Python 3.11, so activate the project environment before running
project scripts:

```bash
source .venv/bin/activate
```

## 4. Confirm Python

Run:

```bash
python --version
```

You want Python 3.11.

Then run:

```bash
python scripts/setup/check_environment.py
```

## 5. Select The Interpreter

If VS Code asks which Python interpreter to use, choose:

```text
.venv/bin/python
```

If it does not ask:

1. Press `Cmd + Shift + P`.
2. Type `Python: Select Interpreter`.
3. Choose the interpreter inside `.venv`.

## 6. Initialize Git

In the VS Code terminal:

```bash
git init
git status
```

Later, when you want to save a checkpoint:

```bash
git add .
git commit -m "Start hackathon workspace"
```

## 7. What Not To Do Yet

Do not install MetaTrader5 yet.
Do not put credentials into files yet.
Do not connect to a broker/platform yet.
Do not place paper orders yet.

First we will build confidence that the local workspace behaves correctly.
