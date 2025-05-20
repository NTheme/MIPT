/*--========================================--*\
    * Author  : NTheme - All rights reserved
    * Created : 25 February 2025, 12:37 AM
    * File    : lib
    * Project : hw1
\*--========================================--*/

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Status {
    NotStarted,
    InProgress,
    Completed,
}

#[derive(Debug, Clone)]
pub struct Task {
    name: String,
    status: Status,
    id: u64,
    description: String,
    assigned_to: Option<String>,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Status::NotStarted => write!(f, "Not Started"),
            Status::InProgress => write!(f, "In Progress"),
            Status::Completed => write!(f, "Completed"),
        }
    }
}

impl Task {
    pub fn new(name: impl Into<String>, id: u64, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::NotStarted,
            id,
            description: description.into(),
            assigned_to: None,
        }
    }

    fn set_status(&mut self, status_from: Status, status_to: Status) -> Result<(), String> {
        match self.status {
            s if s == status_from => {
                self.status = status_to;
                Ok(())
            }
            _ => Err(format!(
                "Unable to apply {:?} status, reason: task is not in {:?} status!",
                status_to, status_from
            )),
        }
    }

    pub fn start(&mut self) -> Result<(), String> {
        self.set_status(Status::NotStarted, Status::InProgress)
    }

    pub fn complete(&mut self) -> Result<(), String> {
        self.set_status(Status::InProgress, Status::Completed)
    }

    pub fn reopen(&mut self) -> Result<(), String> {
        self.set_status(Status::Completed, Status::InProgress)
    }

    pub fn throw(&mut self) -> Result<(), String> {
        self.set_status(Status::InProgress, Status::NotStarted)
    }

    pub fn fast_complete(&mut self) -> Result<(), String> {
        self.set_status(Status::NotStarted, Status::Completed)
    }

    pub fn undone(&mut self) -> Result<(), String> {
        self.set_status(Status::Completed, Status::NotStarted)
    }

    pub fn assign_to(&mut self, user: impl Into<String>) -> Result<(), String> {
        match self.status {
            Status::Completed => {
                Err("Unable to assign task, reason: task has already been completed!".to_string())
            }
            _ => {
                self.assigned_to = Some(user.into());
                Ok(())
            }
        }
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_status(&self) -> Status {
        self.status
    }

    pub fn get_id(&self) -> u64 {
        self.id
    }

    pub fn get_description(&self) -> &String {
        &self.description
    }

    pub fn get_assigned_to(&self) -> Option<&String> {
        self.assigned_to.as_ref()
    }

    pub fn get_assigned_to_lowercased(&self) -> Option<String> {
        self.assigned_to.as_ref().map(|s| s.to_lowercase())
    }

    pub fn get_assigned_to_or_guest(&self) -> String {
        self.assigned_to.clone().unwrap_or("Guest".to_string())
    }

    pub fn is_assigned_to(&self) -> Result<&String, String> {
        self.assigned_to.as_ref().ok_or("Guest".to_string())
    }

    pub fn reassign(&mut self, new_user: impl Into<String>) -> Result<(), String> {
        let old_user = self.is_assigned_to()?;
        let new_user = new_user.into();
        println!(
            "WARNING! Changing assignee from {:?} to {:?}...",
            old_user, new_user
        );
        self.assigned_to = Some(new_user);
        Ok(())
    }
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "---Task {}---\n ID: {}\n Status: {}\n Description: {}\n",
            self.name, self.id, self.status, self.description
        )
    }
}
