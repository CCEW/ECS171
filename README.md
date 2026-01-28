# ECS171

Install VS Code as well as the *Dev Containers* extension. To install the extension go to Extensions in VS Code or (Ctrl + shift + x).
### Prerequisites
- Docker Desktop
- VS Code
- VS Code Dev Containers extension

Install Docker
-
If you do not have Docker installed, download and install Docker Desktop on your system:

- Mac: https://docs.docker.com/desktop/setup/install/mac-install/
- Windows: https://docs.docker.com/desktop/setup/install/windows-install/

Run the following command to verify your installation:
```bash
user@host$ docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.

...
```

 Clone
-
Finally, clone the repository on your personal computer.

In VS Code, simply go to the “Source Control” tab and click on “Clone Repository”. Follow the instructions.


Open project in container
-
In VS Code, open the project using “Open Folder…”.

Assuming that you properly installed the “Dev Containers” extension, VS Code will automatically prompt you to reopen the folder in a container: a small dialog will appear in the bottom right corner offering you to “Reopen in Container.”

**Note:**
The first run may take a few minutes while VS Code builds a new Docker container based of the provided one. Subsequent runs should be instantaneous.''

Download data
-
Download the datasets from Kaggle as zip file:
1. https://www.kaggle.com/datasets/henryhan117/sp-500-historical-data?resource=download
2. https://www.kaggle.com/datasets/innacampo/s-and-p-500-stocks-daily-historical-data-10-years

Place the zips inside their respective folders (historical_data and SP500_10Y). Then, extract everything from the zip, you should have the following:
```bash
data/
    raw/
        historical_data/ SPX.csv
        SP500_10Y/ several csv files
```

## Editing
**IMPORTANT: DO NOT WORK IN MAIN BRANCH**

Use the Source Control tab in VS Code to create a new branch, below Repositories click where it says main, and then choose ***create new branch with YOUR NAME***.

***Adding changes*** to your branch *(Make sure you are working in your branch by testing **git branch** in the terminal)*:
```
Use the “Source Control” tab in VS Code to add the *file* you worked on to the staged area (“Stage Changes” button in the “CHANGES” panel), and commit. Then click “Push” in the “GRAPH” panel for the commit to be pushed to your branch in GitHub.
```
If you do not want to add a file to your girthub branch add it to **.gitignore**