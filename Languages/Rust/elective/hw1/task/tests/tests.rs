/*--========================================--*\
    * Author  : NTheme - All rights reserved
    * Created : 17 March 2025, 4:46 AM
    * File    : tests.rs
    * Project : Rust
\*--========================================--*/
use task::Status;
use task::Task;

#[test]
fn test_create() {
    let task = Task::new("TestTask", 12, "Some description");
    assert_eq!(task.get_name(), "TestTask");
    assert_eq!(task.get_id(), 12);
    assert_eq!(task.get_description(), "Some description");
    assert_eq!(task.get_status(), Status::NotStarted);
    assert_eq!(task.get_assigned_to(), None);
}

#[test]
fn test_valid_transition() {
    let mut task = Task::new("TestTask", 12, "Some description");
    assert!(task.start().is_ok());
    assert_eq!(task.get_status(), Status::InProgress);
    assert!(task.complete().is_ok());
    assert_eq!(task.get_status(), Status::Completed);
    assert!(task.reopen().is_ok());
    assert_eq!(task.get_status(), Status::InProgress);
    assert!(task.throw().is_ok());
    assert_eq!(task.get_status(), Status::NotStarted);
    assert!(task.fast_complete().is_ok());
    assert_eq!(task.get_status(), Status::Completed);
    assert!(task.undone().is_ok());
    assert_eq!(task.get_status(), Status::NotStarted);
}

#[test]
fn test_invalid_transition() {
    let mut task = Task::new("TestTask", 12, "Some description");
    assert!(task.throw().is_err());
    assert!(task.complete().is_err());
    assert!(task.undone().is_err());
    assert!(task.reopen().is_err());

    task.start().unwrap();
    assert!(task.start().is_err());
    assert!(task.fast_complete().is_err());
    assert!(task.undone().is_err());
    assert!(task.reopen().is_err());

    task.complete().unwrap();
    assert!(task.start().is_err());
    assert!(task.fast_complete().is_err());
    assert!(task.throw().is_err());
    assert!(task.complete().is_err());
}

#[test]
fn test_valid_assign_to() {
    let mut task = Task::new("TestTask", 12, "Some description");
    assert!(task.assign_to("Name").is_ok());
    let expected = "Name".to_string();
    assert_eq!(task.get_assigned_to(), Some(&expected));
}

#[test]
fn test_invalid_assign_to() {
    let mut task = Task::new("TestTask", 12, "Some description");
    task.start().unwrap();
    task.complete().unwrap();
    assert!(task.assign_to("Name").is_err());
}

#[test]
fn test_get_assigned_to_lowercase() {
    let mut task = Task::new("TestTask", 12, "Some description");
    task.assign_to("Name").unwrap();
    let user_up = task.get_assigned_to_lowercased();
    assert_eq!(user_up, Some("name".to_string()));
}

#[test]
fn test_get_assigned_to_or_guest() {
    let mut task = Task::new("TestTask", 12, "Some description");
    assert_eq!(task.get_assigned_to_or_guest(), "Guest");
    task.assign_to("Name").unwrap();
    assert_eq!(task.get_assigned_to_or_guest(), "Name");
}

#[test]
fn test_assigned_to_result() {
    let mut task = Task::new("TestTask", 12, "Some description");
    assert!(task.is_assigned_to().is_err());

    task.assign_to("Name").unwrap();
    let user = task.is_assigned_to().unwrap();
    assert_eq!(user, "Name");
}

#[test]
fn test_reassign() {
    let mut task = Task::new("TestTask", 12, "Some description");
    task.assign_to("Name1").unwrap();
    assert!(task.reassign("Name2").is_ok());
    let expected = "Name2".to_string();
    assert_eq!(task.get_assigned_to(), Some(&expected));
}
