from . import agenda, directory, history, internal, meetings, team

routers = [
    meetings.router,
    agenda.router,
    history.router,
    team.router,
    internal.router,
    directory.router,
]
