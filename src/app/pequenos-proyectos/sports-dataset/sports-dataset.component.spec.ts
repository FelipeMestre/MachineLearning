import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SportsDatasetComponent } from './sports-dataset.component';

describe('SportsDatasetComponent', () => {
  let component: SportsDatasetComponent;
  let fixture: ComponentFixture<SportsDatasetComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SportsDatasetComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SportsDatasetComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
