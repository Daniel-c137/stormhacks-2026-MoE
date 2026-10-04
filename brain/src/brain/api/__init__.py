from . import agenda, auth, directory, history, internal, meetings, team

routers = [
    auth.router,
    meetings.router,
    agenda.router,
    history.router,
    team.router,
    internal.router,
    directory.router,
]
