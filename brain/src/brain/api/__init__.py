from . import agenda, history, internal, meetings, team

routers = [meetings.router, agenda.router, history.router, team.router, internal.router]
