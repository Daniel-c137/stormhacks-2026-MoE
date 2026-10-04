from . import directory, history, internal, meetings, team

routers = [meetings.router, history.router, team.router, internal.router, directory.router]
