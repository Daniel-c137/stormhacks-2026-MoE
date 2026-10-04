from . import accounts, agenda, auth, directory, history, internal, meetings, team

routers = [
    auth.router,
    meetings.router,
    agenda.router,
    history.router,
    team.router,
    accounts.router,
    internal.router,
    directory.router,
]
